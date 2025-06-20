from typing import Optional, Tuple, List, Dict, Set
from collections import defaultdict
import numpy as np 
import networkx as nx

from multi_astar.utils import DistanceMap, adjacent
from envs.env import Agent, NextAction, Waypoint, NEIGHBORS


class DistanceGraph(DistanceMap):
    def __init__(self, agents: List[Agent], env_height: int, env_width: int):
        super().__init__(agents, env_height, env_width)
        self.graph = nx.Graph()
        self.lookup = dict()
        self.blocked_cells = defaultdict(set)
        # self.reset()

    def reset(self):
        self.blocked_cells = defaultdict(set)
        for ag in self.agents:
            self.blocked_cells[self.lookup[ag.initial_position]].add(ag.handle)

    def update_blocked_cell(self, agent, new_cell, old_cell=None):
        if old_cell is None:
            old_cell = next((cell for cell, ags in self.blocked_cells.items() if agent in ags), None)
        if old_cell is not None:
            old_cell = self.lookup[old_cell]
            if agent in self.blocked_cells[old_cell]:
                self.blocked_cells[old_cell].remove(agent)
                if len(self.blocked_cells[old_cell]) == 0:
                    del self.blocked_cells[old_cell]
        new_cell = self.lookup[new_cell]
        self.blocked_cells[new_cell].add(agent)
        assert (sum(map(len, self.blocked_cells.values())) == len(self.agents))


    def calc_dist_map(self, maze):
        h = maze.shape[0]
        w = maze.shape[1]
        self.lookup = {
            (i, j): (i, j)
            for i in range(h)
            for j in range(w)
            if maze[i, j] == 0
        }
        self.graph = nx.Graph([  # edge list
            ((i,j), (i+d_i,j + d_j)) 
            for i, j in self.lookup.keys()
            for d_i, d_j in NEIGHBORS
            if h > i + d_i >= 0 and w > j + d_j >= 0 \
                and maze[i+d_i, j+d_j] == 0
        ])
        self.routes = {}
        # remove all nodes who's degree is exactly 2
        remove_nodes = list(
            node 
            for node, deg in self.graph.degree 
            if deg == 2)
        while remove_nodes:
            current = remove_nodes.pop()
            n1, n2 = self.graph[current].keys()
            if self.graph.degree[n1] != 2 and self.graph.degree[n2] != 2:
                # only join nodes that have exactly 2-degree neighbors
                continue
            new_gr = n1
            if n1 not in self.routes:
                prev = [n1]
            else:
                if self.graph.degree[n1] >= 3:
                    if self.graph.degree[n2] < 3:
                        # merge node to n2
                        # switch n1 and n2 
                        n1, n2 = n2, n1
                    else:
                        continue
                prev = self.routes.pop(n1)
                if adjacent(current, prev[0]):
                    # the current node is connected to the first element of the route 
                    # reverse the route, and append the current node to the end
                    prev = prev[::-1]
                else:
                    # check if joining failed
                    assert adjacent(current, prev[-1]), "Distance graph: build error"

            self.graph.remove_node(current)
            self.graph.add_edge(n1, n2)

            if current not in self.routes:
                self.routes[new_gr] = prev + [current]
            else:
                self.routes[new_gr] = prev + self.routes[current]
                del self.routes[current]
            for node in self.routes[new_gr]:
                self.lookup[node] = new_gr

        # block cells for the agent
        self.reset()
        return super().calc_dist_map(maze)


def is_switch_for_grid(dist_map:DistanceGraph):
    grid = np.zeros_like(dist_map.map)
    for n, deg in dist_map.graph.degree:
        if deg > 2:
            grid[n] = 1
    
    return grid


def get_valid_move_actions_(
                            agent_position: Tuple[int, int],
                            graph: nx.graph, width:int, height:int) -> Set[NextAction]:
    neighbors = set()
    assert(0 == 1)
    agent_pos = agent_position[0] * width + agent_position[1]
    for new_position in NEIGHBORS: # Adjacent squares

        # Get node position
        node_position = (agent_position[0] + new_position[0], agent_position[1] + new_position[1])

        # check if valid node
        if node_position[0] >= height or node_position[0] < 0 or node_position[1] >= width or node_position[1] < 0:
            continue

        new_pos = node_position[0] * width + node_position[1]
        # Make sure walkable terrain
        if new_pos not in graph[agent_pos]:
            continue
        
        neighbors.add(node_position)
    return neighbors

def find_next_k_switch(distance_map: DistanceGraph, max_depth: Optional[int] = None, agent_handle: Optional[int] = None, nr_of_switches: Optional[int] = None) \
    -> Dict[int, Optional[List[Waypoint]]]:
    """
    Computes the agent's path to the next switch.
    The paths are derived from a `DistanceGraph`.

    Parameters
    ----------
    distance_map : reference to the distance_map
    max_depth : max path length, if the shortest path is longer, it will be cutted
    agent_handle : if set, the shortest for agent_handle will be returned , otherwise for all agents

    Returns
    -------
        Dict[int, Optional[List[Waypoint]]]

    """
    paths_found = dict()

    def _path_to_target(agent):
        agent_handle = agent.handle

        if agent.position == agent.target:
            paths_found[agent_handle] = None
            return
        else:
            position = agent.position
        
        paths_found[agent_handle] = []
        depth = 0
        distance = np.inf
        while (position != agent.target and (max_depth is None or depth < max_depth)):

            next_actions = get_valid_move_actions_(position, distance_map.graph, distance_map.env_width)
            best_next_action = None
            for next_position in next_actions:
                # distance map - contains the distance from each node in the grid to each agent target 
                next_action_distance = distance_map.get()[
                    agent_handle, 
                    next_position[0], 
                    next_position[1]]
                if next_action_distance < distance:
                    best_next_action = next_position
                    distance = next_action_distance
            
            if best_next_action is None:
                # no possible actions
                # no path found
                paths_found[agent_handle] = None
                return

            # add current cell to the predicted path
            paths_found[agent_handle].append(Waypoint(position))
            depth += 1

            # choose the best action based on which next cell is the nearest to the target (check: what type of distance is it using)
            # update position
            position = best_next_action
        if max_depth is None or depth < max_depth:
            paths_found[agent_handle].append(Waypoint(position))


    if agent_handle is not None:
        _path_to_target(distance_map.agents[agent_handle])
    else:
        for agent in distance_map.agents:
            _path_to_target(agent)

    return paths_found


class GraphPredictor:
    IDX_DIST_TO_TARGET = 4
    IDX_DIST_TO_SWITCH = 3
    def __init__(self, max_depth: int = 20, nr_of_switches: Optional[int]=None, 
                agent_priority_list: Optional[List[int]] = None,
                verify_if_cell_occupied: Optional[bool]=False):
        self.max_depth = max_depth
        self.env = None
        self.verify_is_cell_occupied = verify_if_cell_occupied
        self.nr_of_switches = nr_of_switches
        self.priority_queue = agent_priority_list
    
    def set_env(self, env):
        self.env = env

    def reset(self, nr_of_switches: Optional[int]=None, 
            agent_priority_list: Optional[List[int]] = None):
        self.nr_of_switches = nr_of_switches
        self.priority_queue = agent_priority_list

    def set_priority_queue(self, agent_priority_list:Optional[List[int]]=None):
        self.priority_queue = agent_priority_list

    def get_shortest_paths_all_agents(self):
        """
        Return the shortest paths for all agents
        """
        distance_map: DistanceGraph = self.env.distance_map
        return find_next_k_switch(
            distance_map, 
            max_depth=self.max_depth, 
            nr_of_switches=self.nr_of_switches
        )

    def _get_agent_predicted_path(self, agent, shortest_path):
        agent_handle = agent.handle
        prediction = np.empty(shape=(self.max_depth + 1, 5))

        if agent.position == agent.target:
            agent_virtual_position = agent.target
        else:
            agent_virtual_position = agent.position

        distance_map = self.env.distance_map.get()

        agent_speed = 1
        times_per_cell = 1
        offset = 0
        prediction[0] = [
            0, #timestep difference
            *agent_virtual_position, # position
            # agent_virtual_direction, # direction
            int(not self.is_switch_grid[agent_virtual_position[0], agent_virtual_position[1]]), # is position a simple cell (not a switch) 
            distance_map[
                agent_handle,         
                agent_virtual_position[0],
                agent_virtual_position[1],
                # agent_virtual_direction
            ] # distance from the target position
        ]

        # if there is a shortest path, remove the initial position
        if shortest_path:
            shortest_path = shortest_path[1:]
            # store the first timestep the current cell is occupied, the agent's new direction, the difference between 
            # the first and last timestep that the agent is occupying the current cell
            self.occupied_at_timesteps[agent_handle, agent_virtual_position[0], agent_virtual_position[1]] = [0, 0]

        # initialize the current position
        new_position = agent_virtual_position

        for index in range(1, self.max_depth + 1):
            # if we're at the target, stop moving until max_depth is reached
            if new_position == agent.target or new_position[0] == -1:
                prediction[index] = [
                    index, 
                    -1, -1, 
                    # new_direction, 
                    0, # agent is removed from the map => distance to next switch is set to 0
                    0 # distance from the target position
                ]
                # visited.add((*new_position, new_direction))
                continue
            # no path found
            if not shortest_path:
                prediction[index] = [
                    index, 
                    *new_position, 
                    # new_direction, 
                    int(not self.is_switch_grid[agent_virtual_position[0], agent_virtual_position[1]]), # is position a simple cell (not a switch)
                    # 1, 
                    distance_map[
                        agent_handle,         
                        new_position[0],
                        new_position[1],
                        # new_direction
                    ] # distance from the target position
                ]
                continue

            if (index + offset) % times_per_cell == 0:
                # the agent arrived at the end of the current cell and enters the next grid cell
                # update current position
                new_position = shortest_path[0].position

                shortest_path = shortest_path[1:]

                # label new_position as occupied at the current timestep 
                # save the agent direction and the difference between the first and last timestep when the agent
                #  is occupying the current position
                self.occupied_at_timesteps[agent_handle, new_position[0], new_position[1]] = [index + offset, 0]
            else:
                # increase the length of when the current position is occupied
                self.occupied_at_timesteps[agent_handle, new_position[0], new_position[1], 1] += 1

            prediction[index] = [
                index, # difference between the timesteps
                *new_position, # position
                # new_direction, # direction
                int(not( self.is_switch_grid[new_position[0], new_position[1]] or new_position == agent.target)), # is position a simple cell
                # 1, # is position a simple cell
                distance_map[
                    agent_handle,         
                    new_position[0],
                    new_position[1],
                    # new_direction
                ] # distance from the target position
            ]

        # compute distance to next switch
        self._calc_next_switch_dist(prediction)
        self._calc_dist_from_target(prediction)
        
        return prediction

    def _calc_next_switch_dist(self, prediction):
        # compute distance to next switch - distance between cells
        # distance fractions NOT allowed
        dist_to_switch = 0
        for _idx in range(self.max_depth, -1, -1):                                
            if prediction[_idx][self.IDX_DIST_TO_SWITCH] > 0:
                # increase accumulated distance to the next switch
                dist_to_switch += prediction[_idx][self.IDX_DIST_TO_SWITCH]
                prediction[_idx][self.IDX_DIST_TO_SWITCH] = dist_to_switch
            else:
                # current position is a switch
                dist_to_switch = 0
        return prediction
    
    def _calc_dist_from_target(self, prediction):
        # compute distance to the target - dynamically from the route
        # distance fractions NOT allowed

        # if an agent arrived at the target once, it's not going to move
        # find the last index where dist_from_target > 0, previous values are also going to be > 0
        dist_to_target = 1
        for _idx in range(self.max_depth, -1, -1):                                
            if prediction[_idx, self.IDX_DIST_TO_TARGET] > 0:
                prediction[_idx, self.IDX_DIST_TO_TARGET] = dist_to_target
                dist_to_target += 1
        return prediction

    def get(self, handle: int = None):
        if handle:
            agents = [self.env.agents[handle]]
        elif self.priority_queue is None:
            agents = self.env.agents
        else:
            agents = [self.env.agents[agent_handle] for agent_handle in self.priority_queue]

        number_of_agents = len(agents)

        self.occupied_at_timesteps =  np.empty((number_of_agents,
                                                self.env.height,
                                                self.env.width,
                                                2))
        self.occupied_at_timesteps.fill(np.nan)
        self.is_switch_grid = is_switch_for_grid(self.env.maze)
        
        shortest_paths = self.get_shortest_paths_all_agents()

        prediction_dict = {}

        for agent in agents:
            agent_handle = agent.handle
            # predicted cells to be rendered
            prediction_dict[agent_handle] = self._get_agent_predicted_path(agent, shortest_paths[agent_handle])

        return prediction_dict, prediction_dict

    def get_occupied_at_timesteps(self):
        return self.occupied_at_timesteps