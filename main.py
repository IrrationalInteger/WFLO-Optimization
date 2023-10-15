
# Init



# Constant Wind Direction

# Wind Speed (m/s)
v = 12




# Objective Function




# Decision Variables

# Placement
# Number




# Constraints


# Grid Size (cell*cell). Each cell is (d*d) where d is the diameter of the wind turbine

g_x = 20
g_y = 20

# Minimum distance between every 2 wind turbines (cells).

min_distance = 1

# Maximum number of turbines is: max_n = floor(x / (L + 1)) * floor(y / (L + 1))

max_n = (g_x//(min_distance+1))*(g_y//(min_distance+1))

# User defined maximum number of turbines

n = 100

# Dead Cells

dead_cells = [(4,4)]






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Optimization")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
