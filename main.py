import matplotlib.pyplot as plt
import numpy as np
import math
import random



# constants
POWER_COEFFICIENT = 0.3
LAND_COEFFICIENT = 0.3 # coeffecient of the cost of the land
POWER_THRESHOLD_COEFFICIENT = 0.5
ALPHA = 0.07
# variables
wind_speed = 12
wind_frequency=[1/36]*36

#constraints
n, m = 20, 20 # grid size n*m
dead_cells = [(1, 2), (3, 4), (5, 6)] # example
WT_max_number = 20

#decision variables
WT_list_length = random.randint(1, WT_max_number+1)
WT_list = generate_random_tuples(WT_list_length,dead_cells, m, n) # example : WT_list = [(7.5, 7.5), (9.5, 3.5), (9.5, 9.5), (6.5, 5.5), (4.5, 2.5), (3.5, 0.5), (7.5, 3.5), (5.5, 7.5), (0.5, 2.5), (4.5, 5.5), (8.5, 5.5)]
print(f"WT_coordinates : {WT_list}")

# used to account for the floating point numbers loss of percision
def are_floats_equal(num1, num2, epsilon=1e-5):
    """Check if two floating-point numbers are equal within a tolerance."""
    return abs(num1 - num2) < epsilon

# calculate the intersiction between the wake borders and the grid borders. Used for drawing purposes only
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
  #print(angle)
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

# populates cone grid for all Wind turbines. cone_grid contains the wind turbine cordinates in whose wake the current wind turbine is located. It also contains the distance between the two.
def calculate_cone_grid(WT_coordinates,width,direction,grid_x,grid_y):
  cone_grid = np.empty((grid_x, grid_y), dtype=object)
  for WT in WT_coordinates:
    cells_inside(WT_coordinates,WT,width,direction,grid_x,grid_y,cone_grid)
  return cone_grid

# draws a grid and with certain properties. marked_cells contains cells to be drawn using a different color
def draw_grid(n, m, marked_cells, start, direction, width):
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

# generates a random list of wind turbine coordinates that satisfy the positional constraints:
# 0<=x<m , 0<=y<n
# if (x,y) is present, all adjacent coordinates are excluded
# (x,y) is not in a dead cell
def generate_random_tuples(list_length, exclusion_list, m , n):
    # Create a set of unique tuples within the specified range and not violating the adjacency constraint
    random_tuples = set()

    def is_valid(x, y):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (x + dx, y + dy) in random_tuples:
                    return False
        return True

    while len(random_tuples) < list_length:
        x = random.randint(0, m-1)
        y = random.randint(0, n-1)
        new_tuple = (x, y)

        if new_tuple not in exclusion_list and is_valid(x, y):
            random_tuples.add(new_tuple)

    # Convert the set of tuples back to a list
    random_list = list(random_tuples)
    random_list = [(r[0]+0.5,r[1]+0.5) for r in random_list]
    return random_list


def satisfies_power_constraint(power_frequency, total_power_no_wake):
  power_threshold = total_power_no_wake*POWER_THRESHOLD_COEFFICIENT
  return all(power > power_threshold for power in power_frequency)

# calculates power of a wind turbine in the existence of wake. It first calculates the reduced wind speed due to the wake effect
def calculate_power(wind_speed, wind_speed_wake, distance, rotor_radius=20):
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
    power_grid[WT_x][WT_y] = calculate_power(wind_speed=wind_speed,wind_speed_wake=power_grid[WT_wake_x][WT_wake_y][1],distance=distance*40)

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
  #print(power_grid)
  total_power = 0
  for WT in WT_coordinates:
    WT_x,WT_y = WT
    WT_x = int(WT_x-0.5)
    WT_y = int(WT_y-0.5)
    total_power += power_grid[WT_x][WT_y][0]
  return total_power

# calculate the operatinal and maintainance cost of the wind farm
def calculate_WT_cost(WT_number):
  exponent = -0.00174 * (WT_number**2)
  result = np.exp([exponent])[0]
  cost = WT_number * (2/3 + (1/3 * result))
  return cost

# calculates the cost of the land used by the wind farm.
def calculate_land_cost(WT_coordinates):
  # Separate the list into two lists: one for x coordinates and one for y coordinates
  x_coords = [x for x, y in WT_coordinates]
  y_coords = [y for x, y in WT_coordinates]

  # Calculate the minimum and maximum values for x and y coordinates
  min_x = min(x_coords)
  max_x = max(x_coords)
  min_y = min(y_coords)
  max_y = max(y_coords)

  Area = (max_x-min_x) * (max_y-min_y)
  cost = LAND_COEFFICIENT * Area
  return cost

# calculate the total cost (land and operational cost of the land farm)
def calculate_total_cost(WT_coordinates):
  WT_number = len(WT_coordinates)
  land_cost = calculate_land_cost(WT_coordinates)
  WT_cost = calculate_WT_cost(WT_number)
  total_cost = land_cost + WT_cost
  return total_cost

# calculates the average total power generated by the wind farm. It calculates the cone grid for each wind direction and use it to calculate the power grid.
# It then calculates the weighted average of total power using wind_frequency (wind direction frequency)
def calculate_average_total_power(WT_coordinates,grid_x,grid_y):
  power_frequency=[0]*36
  for idx in range(len(wind_frequency)):
    #calculate cone_grid
    cone_grid = calculate_cone_grid(WT_coordinates, 7, idx * 10, grid_x,grid_y)
    power_frequency[idx] = calculate_total_power(cone_grid,WT_coordinates)
  #print(power_frequency)
  average_total_power = sum(power_frequency * wind_frequency for power_frequency, wind_frequency in zip(power_frequency, wind_frequency))
  if(satisfies_power_constraint(power_frequency, POWER_COEFFICIENT * WT_list_length * (wind_speed**3))):
    print("The solution satisifies the power constraint")
  else:
    print("The solution does not satisify the power constraint")
  return average_total_power

# calculates fitness value of the solution
def objective_function(WT_coordinates,grid_x,grid_y):
  average_total_power = calculate_average_total_power(WT_coordinates,grid_x,grid_y)
  print(f"Average total power : {average_total_power}")
  print(f"Average total power with no wake : {POWER_COEFFICIENT * WT_list_length * (wind_speed**3)}")
  total_cost = calculate_total_cost(WT_coordinates)
  print(f"Total cost : {total_cost}")
  fitness_value = total_cost / average_total_power
  return fitness_value

print(f"fitness value : {objective_function(WT_list,m,n)}")
