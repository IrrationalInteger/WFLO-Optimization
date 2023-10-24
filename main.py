

import numpy as np
import math
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time



# generates a random list of wind turbine coordinates that satisfy the positional constraints:
# 0<=x<m , 0<=y<n
# if (x,y) is present, all adjacent coordinates are excluded
# (x,y) is not in a dead cell
def generate_random_tuples(list_length, exclusion_list, m , n):
    # Create a set of unique tuples within the specified range and not violating the adjacency constraint
    random_tuples = set()

    def is_valid(x, y):
        for dx in [-3, -2, -1, 0, 1, 2, 3]:
            for dy in [-3, -2, -1, 0, 1, 2, 3]:
                if (x + dx, y + dy) in random_tuples:
                    return False
        return True

    i_max = m * n
    i = 0
    while len(random_tuples) < list_length and i < i_max:
        x = random.randint(0, m-1)
        y = random.randint(0, n-1)
        new_tuple = (x, y)

        if new_tuple not in exclusion_list and is_valid(x, y):
            random_tuples.add(new_tuple)
            i=0
        i+=1
    # Convert the set of tuples back to a list
    random_list = list(random_tuples)
    random_list = [(r[0]+0.5,r[1]+0.5) for r in random_list]
    return random_list

# constants
POWER_COEFFICIENT = 0.3
POWER_THRESHOLD_COEFFICIENT = 0.935
ALPHA = 0.1
# variables
wind_speed = 12
wind_frequency=[1/36]*36

#constraints
n, m = 15,15 # grid size n*m
dead_cells = [(2,2),(12,2),(2,12),(12,12)] # no turbines can be placed in these cells
WT_max_number = math.ceil(m/4)*math.ceil(n/4) # user_defined
MAX_WT_number = math.ceil(m/4)*math.ceil(n/4) # defined by the grid size and spacing constraints
assert WT_max_number <= MAX_WT_number

#decision variables
WT_list_length =  1 # or WT_max_number/2

WT_list = generate_random_tuples(WT_list_length,dead_cells, m, n) # example : WT_list = [(7.5, 7.5), (9.5, 3.5), (9.5, 9.5), (6.5, 5.5), (4.5, 2.5), (3.5, 0.5), (7.5, 3.5), (5.5, 7.5), (0.5, 2.5), (4.5, 5.5), (8.5, 5.5)]
#print(f"WT_coordinates : {WT_list}")

# used to account for the floating point numbers loss of precision
def are_floats_equal(num1, num2, epsilon=1e-5):
    """Check if two floating-point numbers are equal within a tolerance."""
    return abs(num1 - num2) < epsilon

# calculate the intersection between the wake borders and the grid borders. Used for drawing purposes only
def calculate_grid_intersection(start,angle,grid_x,grid_y):
  #angle = angle % 360
  if(angle == 0):
    return (grid_x,start[1])
  if(angle == 90):
    return (start[0],grid_y)
  if(angle == 180):
    return (0,start[1])
  if(angle == 270):
    return (start[0],0)
  intersection_points = []
  angle = np.deg2rad(angle)
  cos_theta = np.cos(angle)
  sin_theta = np.sin(angle)

  t = (grid_x - start[0]) / cos_theta
  y = start[1] + (t * sin_theta)
  if(t>0 and y>0-1e-5 and y < grid_y+1e-5):
    intersection_points.append((grid_x,y))

  t = (0 - start[0]) / cos_theta
  y = start[1] + (t * sin_theta)
  if(t>0 and y>0-1e-5 and y < grid_y+1e-5):
    intersection_points.append((0,y))

  t = (grid_y - start[1]) / sin_theta
  x = start[0] + (t * cos_theta)
  if(t>0 and x>0-1e-5 and x < grid_y+1e-5):
    intersection_points.append((x,grid_y))

  t = (0 - start[1]) / sin_theta
  x = start[0] + (t * cos_theta)
  if(t>0 and x>0-1e-5 and x < grid_y+1e-5):
    intersection_points.append((x,0))

  return intersection_points[0]


# returns a boolean that indicates if a point is located inside the wake area.
def is_point_inside(point,start,width,direction):
  if(point == start):
    return True
  vector_origin_point = np.array([point[0]-start[0],point[1]-start[1]])
  vector_origin_point_normalized = vector_origin_point / np.linalg.norm(vector_origin_point)

  direction_radian = np.deg2rad(direction)
  vector_wind_direction = np.array([np.cos(direction_radian),np.sin(direction_radian)])
  vector_wind_direction_normalized = vector_wind_direction / np.linalg.norm(vector_wind_direction)


  cos_angle = np.dot(vector_origin_point_normalized, vector_wind_direction_normalized)
  cos_angle = np.clip(cos_angle, -1.0, 1.0)

  angle = np.arccos(cos_angle)
  angle = np.rad2deg(angle)
  if angle > width:
    return False
  return True

# populates cone_grid for a certain wind turbine. It calculates which wind turbines are located inside the wake area of a certain wind turbine.
def cells_inside(WT_coordinates,start,width,direction,grid_x,grid_y,cone_grid):
  WT_coordinates_copy = [x for x in WT_coordinates if x != start]
  for WT in WT_coordinates_copy:
    x,y = WT
    x = int(x-0.5)
    y = int(y-0.5)
    corners = [(x,y), (x+1,y),(x,y+1),(x+1,y+1),(x+0.5,y+0.5)]
    is_inside = False
    for corner in corners:
      is_inside = is_point_inside(corner,start,width,direction)
      if(is_inside):
        distance = math.dist([x+0.5,y+0.5],list(start))
        if(cone_grid[x][y] != None):
          if(distance < cone_grid[x][y][1]):
            cone_grid[x][y] = (start,distance)
        else:
          cone_grid[x][y] = (start,distance)
        break

# populates cone grid for all Wind turbines. cone_grid contains the wind turbine coordinates in whose wake the current wind turbine is located. It also contains the distance between the two.
def calculate_cone_grid(WT_coordinates,width,direction,grid_x,grid_y):
  cone_grid = np.empty((grid_x, grid_y), dtype=object)
  for WT in WT_coordinates:
    cells_inside(WT_coordinates,WT,width,direction,grid_x,grid_y,cone_grid)
  return cone_grid

# draws a grid and with certain properties. marked_cells contains cells to be drawn using a different color
def draw_grid_with_direction(n, m, marked_cells, start, direction, width):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create the grid
    ax.set_xticks(np.arange(0, m, 1))
    ax.set_yticks(np.arange(0, n, 1))
    plt.xlim(0, m)
    plt.ylim(0, n)

    # Draw grid lines
    ax.grid(which='both')

    # Mark the required cells. Here we no longer need to flip the y coordinate.
    for cell in marked_cells:
        rect = plt.Rectangle((cell[0], cell[1]), 1, 1, color='gray') # cell input is (row, column)
        ax.add_patch(rect)

    # Draw lines from start to end1 and end2
    end1 = calculate_grid_intersection(start,direction + width,m,n)   # (x1, y1)
    end2 = calculate_grid_intersection(start,direction - width,m,n)   # (x2, y2)
    end3 = calculate_grid_intersection(start,direction,m,n)

    ax.plot([start[0], end1[0]], [start[1], end1[1]], 'ro-')
    ax.plot([start[0], end2[0]], [start[1], end2[1]], 'ro-')
    ax.plot([start[0], end3[0]], [start[1], end3[1]], 'g--')

    # Show the plot. The command to invert the y-axis is removed.
    plt.show()

    return end1,end2

def draw_grid(n, m, solution):
    # Create a figure and axis
    fig, ax = plt.subplots()
    plt.subplot(nrows=1, ncolumns=2, index=2)
    # Create the grid
    ax.set_xticks(np.arange(0, m, 1))
    ax.set_yticks(np.arange(0, n, 1))
    plt.xlim(0, m)
    plt.ylim(0, n)

    # Draw grid lines
    ax.grid(which='both')

    # Mark the required cells. Here we no longer need to flip the y coordinate.
    for cell in solution:
        rect = plt.Rectangle((cell[0]-0.5, cell[1]-0.5), 1, 1, color='gray') # cell input is (row, column)
        ax.add_patch(rect)

    # Show the plot. The command to invert the y-axis is removed.
    plt.show()

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

# Adds a new turbine while respecting the spacing distance and dead cells
def add_new_WT(solution, exclusion_list, m , n):
  for i in range(len(solution)):
    solution[i] = (solution[i][0] - 0.5, solution[i][1] - 0.5)
  def is_valid(x, y):
        for dx in [-3,-2,-1, 0, 1,2,3]:
            for dy in [-3,-2,-1, 0, 1,2,3]:
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


# Linear scheduling formula
def calculate_T_linear(T, step):
  return T - step
# Geometric scheduling formula
def calculate_T_geometric(T, factor):
    return factor * T
# SA parameters
T_initial = 500 # Initial temperature of annealing
T_final = 0 # Final temperature of annealing
iteration_per_T = 2 # The number of solutions generated per temperature
i_max = 500 # Artificial stopping condition
factor = 1 # Factor used for decreasing temperature. Used as step for linear and factor for geometric.
fitness_value_scaling_factor = 10000000 # Scaling of fitness for temperature
calculate_T = calculate_T_linear # Choice of scheduling function


def simulated_annealing():
  current_solution = WT_list # Initial solution set to the randomized layout
  current_fitness,__,_ = objective_function(current_solution,m,n) # Initial fitness
  best_solution = current_solution # Best solution so far
  best_fitness = current_fitness # Best fitness so far
  T_current = T_initial # Temperature init
  objective_vs_N = [float('inf')]*(WT_max_number+1) # Objective vs Number of Turbines for plotting
  power_vs_N = [float('inf')]*(WT_max_number+1) # Power vs Number of Turbines for plotting
  objective_vs_I = [] # Objective vs iterations for plotting
  optimal_objective_vs_I = []# Optimal Objective vs iterations for plotting
  i = 0
  probability = 0
  while T_current > T_final and i <= i_max:
    for j in range(iteration_per_T):
      # Generate new solution
      new_solution = generate_neighbour_solution(current_solution, dead_cells, m, n)
      new_fitness,new_power,satisfies = objective_function(new_solution,m,n)
      objective_vs_N[len(current_solution)] = new_fitness if objective_vs_N[len(current_solution)]> new_fitness else objective_vs_N[len(current_solution)]
      power_vs_N[len(current_solution)] = new_power if power_vs_N[len(current_solution)]> new_power else power_vs_N[len(current_solution)]
      # Calculate the delta in fitness
      objective_function_change = new_fitness - current_fitness
      if objective_function_change < 0: # If negative, then improvement: keep this solution
        current_solution = new_solution
        current_fitness = new_fitness
      else: # Else randomize a number from 0 to 1
        random_number = random.uniform(0, 1)
        # Scale fitness to range of temperature
        objective_function_change_scaled = objective_function_change * fitness_value_scaling_factor
        # Boltzmann-Gibbs calculation for probability
        probability = math.exp(-objective_function_change_scaled/T_current)
        if random_number < probability: # If greater than random_number then keep
          current_solution = new_solution
          current_fitness = new_fitness
      if current_fitness < best_fitness and satisfies: # Accept the new solution as optimal iff it has better fitness and also satisifies the power constraint
        best_solution = current_solution
        best_fitness = current_fitness
        ax2.set_title("Best Solution"
                      "\nFitness:" + str(round(best_fitness, 8)))
      # Draw new generation
      plt.pause(0.1)  # Pause to view the updated plot
      ax1.set_title('Generated Solution'
                    '\nTemperature:' + str(T_current)+
                    "\nIteration:"+str(j)+
                    "\nProbability:"+str(round(probability,3))+
                    "\nFitness:"+str(round(current_fitness,8)))
      update_grid(grid1,cax1,current_solution,best_solution,True)
      update_grid(grid2,cax2,None,best_solution,False)

      print(current_fitness)
      print(current_solution)
      objective_vs_I.append(current_fitness)
      optimal_objective_vs_I.append(best_fitness)
    T_current = calculate_T(T_current,factor)
    i = i + 1
    print(f"T_current : {T_current}")
  return best_solution,best_fitness,objective_vs_N,power_vs_N,objective_vs_I,optimal_objective_vs_I

# Initializes grid for layout drawing
def draw_simulation():
    # Create a white grid
    global grid1
    grid1 = np.ones((n, m,4))
    global grid2
    grid2 = np.ones((n, m,4))
    global ax1,ax2,cax1,cax2
    # Create a figure with two subplots (side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Create the initial plots with a color map
    cax1 = ax1.imshow(grid1, cmap='bwr', vmin=0, vmax=1)
    cax2 = ax2.imshow(grid2, cmap='bwr', vmin=0, vmax=1)
    ax1.set_xticks(np.arange(n))
    ax2.set_xticks(np.arange(n))
    ax1.set_yticks(np.arange(m))
    ax2.set_yticks(np.arange(m))
    ax1.invert_yaxis()
    ax2.invert_yaxis()

    print(np.arange(m)[::-1])


    # Function to add gridlines to an axis
    def add_gridlines(ax):
        ax.set_xticks(np.arange(-.5, m, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
        ax.tick_params(which="both", length=0)

    # Add gridlines
    add_gridlines(ax1)
    add_gridlines(ax2)
    ax1.set_title('Generated Solution\nTemperature:\nIteration:')
    ax2.set_title('Best Solution')


    plt.show(block=False)
    # Function to update grid with new coordinates

# Updates grid with a new turbine layout
def update_grid(grid, cax, coords_red, coords_blue,blue_trans):
        # Reset the grid to all white with full opacity
        grid[:, :, :3] = 1  # All pixels white
        grid[:, :, 3] = 0  # Full opacity (no transparency)
        for coord in dead_cells:
            y, x = int(coord[0]), int(coord[1])
            if 0 <= x < n and 0 <= y < m:
                grid[x, y, :3] = [0.2,0.2,0.2]
                grid[x, y, 3] = 1
        # Define the RGBA values for grey and blue with 50% transparency
        grey_color = np.array([0.5, 0.5, 0.5, 0.8])
        red_color = np.array([1, 0, 0, 0.4])
        blue_color = np.array([0, 0, 1, 0.6])

        # Create a copy of the grid representing the current state
        current_colors = np.copy(grid)

        # Apply the new coordinates for grey
        if(coords_red!=None):
            for coord in coords_red:
              y, x = int(coord[0]), int(coord[1])
              if 0 <= x < n and 0 <= y < m:
                 grid[x, y, :3] = red_color[:3]
                 grid[x, y, 3] = 1

        # Apply the new coordinates for blue
        if(coords_blue!=None):
            for coord in coords_blue:
                y, x = int(coord[0]), int(coord[1])
                if 0 <= x < n and 0 <= y < m:
                    grid[x, y, 3] = 1

                    if(np.array_equal(grid[x,y,:3],red_color[:3])):
                        grid[x, y, :3] = [1,0,0]
                        grid[x, y, 3] = 0.5

                    else:
                        if(blue_trans):
                            grid[x, y, 3] = 0.3
                        grid[x, y, :3] = blue_color[:3]

        cax.set_data(grid)  # Update plot data # Update plot data

# Plots the number of turbines against the power and objective function during annealing
def draw_number_of_turbines_against_power_and_objective(power_data,objective_data):

    # Clean from float.inf
    temp_objective_vs_N = [t for t in objective_data if t != float('inf')]
    temp_power_vs_N = [t for t in power_data if t != float('inf')]
    bounds = [min(temp_objective_vs_N), max(temp_objective_vs_N), min(temp_power_vs_N), max(temp_power_vs_N)]
    objective_data = [(index, t) for index, t in enumerate(objective_data) if t != float('inf')]
    power_data = [(index, t) for index, t in enumerate(power_data) if t != float('inf')]

    power_x, power_y = zip(*power_data)
    objective_x, objective_y = zip(*objective_data)

    # Create a figure and a subplot
    fig, ax1 = plt.subplots()

    # Plot the first line with 'power_y' on the left y-axis
    color = 'tab:red'
    ax1.set_xlabel('Number of turbines')
    ax1.set_ylabel('Power', color=color)
    ax1.plot(power_x, power_y, color=color, label='Power')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(bounds[2], bounds[3])  # Set the limits of the left y-axis

    # Instantiate a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Objective', color=color)
    ax2.plot(objective_x, objective_y, color=color, label='Objective',
             linestyle='--')  # We use a dashed line for the second line
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(bounds[0], bounds[1])  # Set the limits of the right y-axis

    # Set the limits of the x-axis
    ax1.set_xlim(0, len(power_data))

    # Add a legend
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

    # Show the plot
    plt.show()

# Plots the number of generated solutions against the objective functino
def draw_iterations_against_solution(objective_data,optimal):

    # Create a figure and a subplot
    fig, ax1 = plt.subplots()
    annealing_iterations = len(objective_data)
    # Plot the first line with 'power_y' on the left y-axis
    color = 'tab:red'
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Optimal Objective Function'if optimal else "Generated Objective Function", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(min(objective_data), max(objective_data))  # Set the limits of the left y-axis
    objective_data = [(index, t) for index, t in enumerate(objective_data)]
    objective_x, objective_y = zip(*objective_data)
    ax1.plot(objective_x, objective_y, color=color, label='Objective')

    # Set the limits of the x-axis
    ax1.set_xlim(0, annealing_iterations)

    # Add a legend
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))

    # Show the plot
    plt.show()


# Test case 1 is run by default

# Uncomment this block for test case 2
#n,m = 15,15
#dead_cells = [(2,2),(12,2),(2,12),(12,12)]
#T_initial = 1000
#factor = 0.95
#calculate_T = calculate_T_geometric

# Uncomment this block for test case 3
#n,m = 20,20
#dead_cells = [(3,2),(4,2),(3,3),(4,3),(15,2),(16,2),(15,3),(16,3),(3,16),(4,16),(3,17),(4,17),(15,16),(16,16),(15,17),(16,17)]
#T_initial = 500
#factor = 1
#calculate_T = calculate_T_linear

# Uncomment this block for test case 4
#n,m = 20,20
#dead_cells = [(3,2),(4,2),(3,3),(4,3),(15,2),(16,2),(15,3),(16,3),(3,16),(4,16),(3,17),(4,17),(15,16),(16,16),(15,17),(16,17)]
#T_initial = 1000
#factor = 0.95
#calculate_T = calculate_T_geometric

# Uncomment this block for test case 5
#n,m = 25,25
#dead_cells = [(5,5),(5,6),(6,5),(6,6),(5,18),(5,19),(6,18),(6,19),(18,5),(19,5),(18,6),(19,6),(18,18),(18,19),(19,18),(19,19),(7,7),(7,6),(7,5),(7,18),(7,19),(18,7),(19,7),(5,7),(6,7),(5,17),(6,17),(7,17),(17,5),(17,6),(17,7),(17,17),(17,18),(17,19),(18,17),(19,17)]
#T_initial = 500
#factor = 1
#calculate_T = calculate_T_linear

# Uncomment this block for test case 6
#n,m = 25,25
#dead_cells = [(5,5),(5,6),(6,5),(6,6),(5,18),(5,19),(6,18),(6,19),(18,5),(19,5),(18,6),(19,6),(18,18),(18,19),(19,18),(19,19),(7,7),(7,6),(7,5),(7,18),(7,19),(18,7),(19,7),(5,7),(6,7),(5,17),(6,17),(7,17),(17,5),(17,6),(17,7),(17,17),(17,18),(17,19),(18,17),(19,17)]
#T_initial = 1000
#factor = 0.95
#calculate_T = calculate_T_geometric


draw_simulation()
time.sleep(3) # Delay to allow grid to properly initialize. May need to rerun code multiple times for it to work

best_solution,best_fitness,objective_vs_N,power_vs_N,objective_vs_I,optimal_objective_vs_I = simulated_annealing()

draw_number_of_turbines_against_power_and_objective(power_vs_N,objective_vs_N)
draw_iterations_against_solution(objective_vs_I,False)
draw_iterations_against_solution(optimal_objective_vs_I,True)

print("Optimal solution:"+str(best_solution))
print("Optimal fitness value:"+str(best_fitness))
