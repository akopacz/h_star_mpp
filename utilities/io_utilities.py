import matplotlib.pyplot as plt
import numpy as np


def answer_for_robot_coordinates(n_robots):
    """
    This function ask to the user for de start and end coordinates of the robots in the problem.

    :param n_robots: Number of robots in the problem
    :return: Coordinates introduced by the user
    """
    coordinates = []
    for i in range(n_robots):
        print(f"Please introduce these coordinates for the robot number {i + 1}")
        x_start = ask_for_coordinate("X coordinate for the start point")
        y_start = ask_for_coordinate("Y coordinate for the start point")
        x_end = ask_for_coordinate("X coordinate for the end point")
        y_end = ask_for_coordinate("Y coordinate for the end point")
        coordinates += [([y_start, x_start], [y_end, x_end])]
    return coordinates


def load_robot_coordinates(filename):
    '''
    This function loads a file taht contains the coordinates that the robots need to calculate the route. This file must have a line per robot, and each line have two coordinates (start and final) separated by ';'. Each coordinate have two components x and y, separated by ','. In summary, each line should look like this:
    x_start,y_start;x_end,y_end

    :param filename: Path to load the file with information
    :return: list of coordinates load from file
    '''
    coordinates = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            l = line.strip()
            if len(l) != 0 and l[0] != '#':
                coords = l.split(';')
                start = coords[0].split(',')
                end = coords[1].split(',')
                coordinates += [([int(start[1]), int(start[0])], [int(end[1]), int(end[0])])]
    return coordinates


def ask_for_coordinate(text):
    """
    Function to help askig for cordinates. Ask for the user with a custom text and check if answer is an integer. Keep asking until a valid coordinate is introduced

    :param text: Text to display when asking for the coordinate
    :return: Integer of the value introduced by the user
    """
    while True:
        try:
            return int(input(f"{text}: "))
        except:
            print("The coordinate must be an integer")


def load_obstacle_matrix(path):
    """
    Loads the matrix that represent the obstacles from a file. The file must contain whitespaces for the free cells and # for the obstacle cells.

    :param path: Path where the file is
    :return: Numpy matrix with 0's on free cells and 1's on cells with obstacles
    """
    with open(path) as f:
        lines = f.read().splitlines()

    matrix = np.zeros(shape=(len(lines), max([len(lines[i]) for i in range(len(lines))])), dtype=int)
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            if lines[i][j] == '#':
                matrix[i, j] = 1

    return matrix


def show_routes(matrix, routes=None, start_end_coords=None, filename="output.png"):
    '''
    Output function that generate a plot that shows the map and its routes (optionally)

    :param matrix: Matrix with the map to draw. 1 for obstacles and 0 for clear cells
    :param routes: (optional) Routes to draw on the map
    :return: None
    '''
    plt.clf()
    plt.imshow(matrix, cmap='binary')

    ax = plt.gca()
    ax.set_xticks(np.arange(-.55, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.55, matrix.shape[0], 1), minor=True)
    # Gridlines based on minor ticks
    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)

    # Remove minor ticks
    ax.tick_params(which='minor', bottom=False, left=False)

    if routes is not None:
        for ind, route in enumerate(routes):
            if route is not None:
                ys, xs = zip(*route)
                ys = list(ys)
                xs = list(xs)
                if start_end_coords is not None:
                    ys.insert(0, start_end_coords[ind][0][0])
                    xs.insert(0, start_end_coords[ind][0][1])
                plt.plot(xs, ys)

    if start_end_coords is not None:
        starts, ends = zip(*start_end_coords)
        for start in starts:
            plt.scatter([start[1] - 0.025], [start[0] - 0.025], marker=".")
        plt.gca().set_prop_cycle(None)
        for end in ends:
            plt.scatter([end[1] - 0.025], [end[0] - 0.025], marker="s")
    plt.savefig(filename)
    # plt.show()


def show_map(matrix, robots=None):
    '''
    Output function that generate a plot that shows the map and the routes of the robots (optionally)

    :param matrix: Matrix with the map to draw. 1 for obstacles and 0 for clear cells
    :param routes: (optional) Robots to draw its routes on the map
    :return: None
    '''
    show_routes(matrix, [robot.route for robot in robots] if robots is not None else None)
