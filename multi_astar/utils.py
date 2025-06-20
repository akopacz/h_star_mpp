from typing import Optional, Tuple, List, Dict, Set
import numpy as np 
import heapq
from envs.env import Agent, NextAction, Waypoint, NEIGHBORS


def adjacent(cell1, cell2):
    return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1]) <= 1


class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    
    def __lt__(self, other):
        return self.position < other.position

def eucl(x, y):
    return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2) 

def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    # dist map related
    width, height = maze.shape
    dist_map = np.empty((width, height), dtype=float)
    dist_map[:] = np.inf
    dist_map[end[0], end[1]] = 0

    # Create start and end node
    start_node = Node(None, tuple(end))
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, tuple(start))
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = [(0, start_node)]
    closed_list = set()
    heapq.heapify(open_list)

    # Loop until you find the end
    while len(open_list) > 0:
        _, current_node = heapq.heappop(open_list)

        if current_node.position in closed_list:
            continue
        closed_list.add(current_node.position)
        dist_map[current_node.position[0], current_node.position[1]] = current_node.g

        # Generate children
        children = []
        for new_position in NEIGHBORS: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] >= width or node_position[0] < 0 or node_position[1] >= height or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            if child.position in closed_list:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = eucl(
                end_node.position,
                child.position
            )
            child.f = child.g + child.h

            # Add the child to the open list
            heapq.heappush(open_list, (child.f, child))
    return dist_map

class DistanceMap:
    def __init__(self, agents: List[Agent], env_height: int, env_width: int):
        self.env_height = env_height
        self.env_width = env_width
        self.distance_map = None
        self.agents_previous_computation = None
        self.reset_was_called = False
        self.agents: List[Agent] = agents
        self.map: Optional[np.ndarray] = None

    def calc_dist_map(self, maze):
        self.map = maze.copy()
        self.distance_map = np.empty((len(self.agents), self.env_height, self.env_width))
        for agent in self.agents:
            self.distance_map[agent.handle] = astar(maze, agent.initial_position, agent.target)

        return self.distance_map

    def get(self):
        return self.distance_map

    def reset(self):
        pass 

    def update_blocked_cell(self, *args, **kwargs):
        pass


def get_valid_move_actions_(
                            agent_position: Tuple[int, int],
                            maze: np.ndarray) -> Set[NextAction]:
    actions = set()
    for new_position in NEIGHBORS: # Adjacent squares

        # Get node position
        node_position = (agent_position[0] + new_position[0], agent_position[1] + new_position[1])

        # Make sure within range
        if node_position[0] >= maze.shape[0] or node_position[0] < 0 or node_position[1] >= maze.shape[1] or node_position[1] < 0:
            continue

        # Make sure walkable terrain
        if maze[node_position[0]][node_position[1]] != 0:
            continue
        
        # TODO: change the None to the action needed to take that step 
        actions.add((None, node_position))
    return actions

def is_switch(transitions:int, return_nr_of_transitions:Optional[bool]=False) -> Tuple[bool, Optional[int]]:
    return True


def find_next_k_switch(distance_map: DistanceMap, max_depth: Optional[int] = None, agent_handle: Optional[int] = None, nr_of_switches: Optional[int] = None) \
    -> Dict[int, Optional[List[Waypoint]]]:
    """
    Computes the agent's path to the next switch.
    The paths are derived from a `DistanceMap`.

    Parameters
    ----------
    distance_map : reference to the distance_map
    max_depth : max path length, if the shortest path is longer, it will be cutted
    agent_handle : if set, the shortest for agent_handle will be returned , otherwise for all agents

    Returns
    -------
        Dict[int, Optional[List[Waypoint]]]

    """
    path_to_next_switch = dict()

    def _path_to_next_switch(agent):
        agent_handle = agent.handle

        if agent.position == agent.target:
            path_to_next_switch[agent_handle] = None
            return
        else:
            position = agent.position
        
        # direction = agent.direction
        path_to_next_switch[agent_handle] = []
        depth = 0
        distance = np.inf
        while (position != agent.target and (max_depth is None or depth < max_depth)):

            next_actions = get_valid_move_actions_(position, distance_map.map)
            best_next_action = None
            for next_action, next_position in next_actions:
                # distance map - contains the distance from each node in the grid to each agent target 
                next_action_distance = distance_map.get()[
                    agent_handle, 
                    next_position[0], 
                    next_position[1]]
                if next_action_distance < distance:
                    best_next_action = next_action, next_position
                    distance = next_action_distance
            
            if best_next_action is None:
                # no possible actions
                # no path found
                path_to_next_switch[agent_handle] = None
                return

            # add current cell to the predicted path
            path_to_next_switch[agent_handle].append(Waypoint(position))
            depth += 1

            # choose the best action based on which next cell is the nearest to the target (check: what type of distance is it using)
            # update position
            position = best_next_action[1]
            # direction = best_next_action[1]
        if max_depth is None or depth < max_depth:
            path_to_next_switch[agent_handle].append(Waypoint(position))


    if agent_handle is not None:
        _path_to_next_switch(distance_map.agents[agent_handle])
    else:
        for agent in distance_map.agents:
            _path_to_next_switch(agent)

    return path_to_next_switch
