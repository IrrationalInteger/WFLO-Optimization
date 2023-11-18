# calculate the intersection between the wake borders and the grid borders. Used for drawing purposes only
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('TkAgg')


def calculate_grid_intersection(start, angle, grid_x, grid_y):
    # angle = angle % 360
    if (angle == 0):
        return (grid_x, start[1])
    if (angle == 90):
        return (start[0], grid_y)
    if (angle == 180):
        return (0, start[1])
    if (angle == 270):
        return (start[0], 0)
    intersection_points = []
    angle = np.deg2rad(angle)
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    t = (grid_x - start[0]) / cos_theta
    y = start[1] + (t * sin_theta)
    if (t > 0 and y > 0 - 1e-5 and y < grid_y + 1e-5):
        intersection_points.append((grid_x, y))

    t = (0 - start[0]) / cos_theta
    y = start[1] + (t * sin_theta)
    if (t > 0 and y > 0 - 1e-5 and y < grid_y + 1e-5):
        intersection_points.append((0, y))

    t = (grid_y - start[1]) / sin_theta
    x = start[0] + (t * cos_theta)
    if (t > 0 and x > 0 - 1e-5 and x < grid_y + 1e-5):
        intersection_points.append((x, grid_y))

    t = (0 - start[1]) / sin_theta
    x = start[0] + (t * cos_theta)
    if (t > 0 and x > 0 - 1e-5 and x < grid_y + 1e-5):
        intersection_points.append((x, 0))

    return intersection_points[0]


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
        rect = plt.Rectangle((cell[0], cell[1]), 1, 1, color='gray')  # cell input is (row, column)
        ax.add_patch(rect)

    # Draw lines from start to end1 and end2
    end1 = calculate_grid_intersection(start, direction + width, m, n)  # (x1, y1)
    end2 = calculate_grid_intersection(start, direction - width, m, n)  # (x2, y2)
    end3 = calculate_grid_intersection(start, direction, m, n)

    ax.plot([start[0], end1[0]], [start[1], end1[1]], 'ro-')
    ax.plot([start[0], end2[0]], [start[1], end2[1]], 'ro-')
    ax.plot([start[0], end3[0]], [start[1], end3[1]], 'g--')

    # Show the plot. The command to invert the y-axis is removed.
    plt.show()

    return end1, end2


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
        rect = plt.Rectangle((cell[0] - 0.5, cell[1] - 0.5), 1, 1, color='gray')  # cell input is (row, column)
        ax.add_patch(rect)

    # Show the plot. The command to invert the y-axis is removed.
    plt.show()
