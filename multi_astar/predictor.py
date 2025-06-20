import numpy as np
from typing import Dict, List, Optional, Tuple

from multi_astar.utils import DistanceMap, find_next_k_switch
from envs.env import Agent, NextAction, Waypoint, NEIGHBORS


def is_switch_for_grid(maze):
    grid = np.zeros_like(maze)
    for x in range(maze.shape[0]):
        for y in range(maze.shape[1]):
            if maze[x, y] == 0:
                nr_neigh = sum(maze[x+n[0], y+n[1]] == 0 for n in NEIGHBORS)
                if nr_neigh > 2:
                    grid[x, y] = 1
    
    return grid

class NextKSwitchPredictor:
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
        distance_map: DistanceMap = self.env.distance_map
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
        
        return prediction

    def _calc_next_switch_dist(self, prediction):
        # compute distance to next switch - distance between cells
        # distance fractions NOT allowed
        dist_to_switch = 0
        for _idx in range(self.max_depth, -1, -1):                                
            if prediction[_idx][4] > 0:
                # increase accumulated distance to the next switch
                dist_to_switch += prediction[_idx][4]
                prediction[_idx][4] = dist_to_switch
            else:
                # current position is a switch
                dist_to_switch = 0
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