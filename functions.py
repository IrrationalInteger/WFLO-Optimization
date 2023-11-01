
# generates a random list of wind turbine coordinates that satisfy the positional constraints:
# 0<=x<m , 0<=y<n
# if (x,y) is present, all adjacent coordinates are excluded
# (x,y) is not in a dead cell
import math
import random
import numpy as np


# Create a set of unique tuples within the specified range and not violating the adjacency constraint
def generate_random_tuples(list_length, exclusion_list, m, n, spacing_distance):
    random_tuples = set()

    def is_valid(x, y):
        for dx in list(range(-spacing_distance,spacing_distance+1)):
            for dy in list(range(-spacing_distance,spacing_distance+1)):
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

