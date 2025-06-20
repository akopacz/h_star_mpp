import numpy as np

def return_path(path, maze):
    moves = {"R": [0, 1],
             "L": [0, -1],
             "D": [1, 0],
             "U": [-1, 0],
             "H": [0, 0]}
    result = ""
    # we update the path of start to end found by A-star serch with every step incremented by 1
    for i in range(1, len(path)):
        error = True
        for move in moves.keys():
            if path[i][0] == path[i-1][0] + moves[move][0] and path[i][1] == path[i-1][1] + moves[move][1]:
                result += move
                error = False
        if error:
            raise Exception("Cannot reproduce path")
    return result, path