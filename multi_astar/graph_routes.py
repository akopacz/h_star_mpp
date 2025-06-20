import numpy as np
import networkx as nx
from operator import itemgetter
from itertools import product
from typing import Dict, List, Optional, Tuple, Set, Generator

from multi_astar.graph_predictor import GraphPredictor, is_switch_for_grid, DistanceGraph
from multi_astar.utils import DistanceMap, adjacent
from envs.env import reverse_dict, NextAction, NEIGHBORS, _get_action_to_move


def get_valid_move_actions_(ag:int,
                            agent_position: Tuple[int, int],
                            dist_graph: DistanceGraph) -> Set[NextAction]:
    agent_pos = dist_graph.lookup[agent_position]
    for node_position in dist_graph.graph[agent_pos]:
        other_agent = dist_graph.blocked_cells.get(node_position, None)  # this should be a set()
        if not other_agent:  # or other_agent == ag:  # do not let the agent to step back to it's original position
            # True if a) None (node_position not in blocked_cells) or b) other_agent is an empty set
            yield node_position


def get_intermediate_positions(position, next_position, distance_map:DistanceGraph) -> Generator:
    if adjacent(position, next_position):
        # cells are adjacent
        yield position
        return 

    super_node = distance_map.lookup[position]
    next_super_node = distance_map.lookup[next_position]
    route = distance_map.routes.get(super_node)
    if route is None:
        route = [position]
        i = 0
    else:
        i = route.index(position)
        if super_node == next_super_node:
            j = route.index(next_position)
            if i < j:
                yield from route[i:j]
            else:
                yield from route[j+1:i+1][::-1]
            return
    next_route = distance_map.routes.get(next_super_node)
    if next_route is None:
        next_route = [next_position]
        j = 0
    else:
        j = next_route.index(next_position)

    for f, n in product(range(-1, len(route) - 1), range(-1, len(next_route) - 1)):  # start iterators from -1, so the last elements are checked first
        if adjacent(route[f], next_route[n]):
            # connection found
            if f < 0:
                # start from the end, append up to this position
                yield from route[i:]
            else:
                yield from route[:i+1][::-1]
            if n < 0:
                # start from the end, append up to this position
                yield from next_route[j+1:][::-1]
            else:
                yield from next_route[:j]
            return
    


def get_shortest_path_from_switch(distance_map: DistanceGraph, \
    max_depth: Optional[int] = None, agent_handle: Optional[int] = None) \
    -> Dict[int, Optional[List[int]]]:
    """
    Computes the agent's path to the next switch.
    The paths are derived from a `DistanceMap`.

    If there is no path (rail disconnected), the path is given as None.
    The agent state (moving or not) and its speed are not taken into account


    Parameters
    ----------
    distance_map : reference to the distance_map
    max_depth : max path length, if the shortest path is longer, it will be cutted
    agent_handle : if set, the shortest for agent_handle will be returned , otherwise for all agents

    Returns
    -------
        Dict[int, Optional[List[Waypoint]]]

    """
    shortest_paths = {}

    def _path_for_agent(agent):
        agent_handle = agent.handle
        shortest_paths[agent_handle] = {}

        target = distance_map.lookup[agent.target]
        if agent.position == agent.target:
            return
        elif target == distance_map.lookup[agent.position]:
            # there is a clear path towards the target
            shortest_paths[agent_handle][0] = list(get_intermediate_positions(agent.position, agent.target, distance_map))
            shortest_paths[agent_handle][0].append(agent.target)
            return
        else:
            position = agent.position
        
        # initialization
        k = 0
        cont_from = {}

        # one step look ahead
        next_actions = get_valid_move_actions_(agent_handle, 
                                               position, 
                                               distance_map)
        for next_position in next_actions:
            part = list(get_intermediate_positions(position, next_position, distance_map))

            # use this as a 1 step look ahead
            # filter out paths that would collide (within the lookahead) with other agents 
            filtered = False
            # for ag in distance_map.agents:
            #     if ag.handle != agent_handle and (ag.position == next_position or ag.position in part):
            #         filtered = True
            #         break
            if not filtered:
                shortest_paths[agent_handle][k] = part.copy()
                cont_from[k] = next_position, len(part) 
                k += 1

        distances_lookup = distance_map.get()
        # k stores the number of different agent paths that will be generated
        for i in cont_from.keys():
            # recover initial position, direction and the depth
            position, depth = cont_from[i]
            # position = prev_action.next_position
            while (position != target and (max_depth is None or depth < max_depth)):
                next_actions = get_valid_move_actions_(
                    agent_handle,
                    position, 
                    distance_map)
                distance = np.inf
                best_next_action = None
                for next_position in next_actions:
                    if next_position not in distance_map.routes:
                        # do not let agents to step back to previous positions - thus, available routes are prioritized over the shortest one
                        # let agents revisit "super-cells" more than one - may be changed later
                        if next_position in shortest_paths[agent_handle][i]:
                            continue
                    # distance map - contains the distance from each node in the grid to each agent target 
                    next_action_distance = distances_lookup[
                        agent_handle, 
                        next_position[0], 
                        next_position[1]
                    ]
                    if next_position in distance_map.routes:
                        first = distance_map.routes[next_position][0]
                        first = distances_lookup[agent_handle, 
                                             first[0], first[1]]
                        last = distance_map.routes[next_position][-1]
                        last = distances_lookup[agent_handle, 
                                             last[0], last[1]]
                        if first < next_action_distance:
                            # step on first, than enter the corridor
                            next_action_distance = first
                        elif last < next_action_distance:
                            next_action_distance = last
                            if next_position == target:
                                best_next_action = next_position
                                # target is found. this is the most optimal action, stop searching
                                break
                            # do not add next_position to the route, break from loop
                    if next_action_distance < distance:
                        best_next_action = next_position
                        distance = next_action_distance
                
                if best_next_action is None:
                    # no possible actions
                    # no path found
                    shortest_paths[agent_handle][i] = None
                    break

                # add current cell to the predicted path
                tmp = tuple(get_intermediate_positions(position, best_next_action, distance_map))
                shortest_paths[agent_handle][i].extend(tmp)
                depth += len(tmp)

                # choose the best action based on which next cell is the nearest to the target (check: what type of distance is it using)
                # update position
                # prev_action = best_next_action
                position = best_next_action
            # if target is reached 
            if position == target:
                # append target to the list of positions
                last = shortest_paths[agent_handle][i][-1]
                if not adjacent(last, agent.target):
                    shortest_paths[agent_handle][i].extend(get_intermediate_positions(position, agent.target, distance_map))
                    last = shortest_paths[agent_handle][i][-1]
                if last != agent.target: 
                    shortest_paths[agent_handle][i].append(agent.target)
    if agent_handle is not None:
        _path_for_agent(distance_map.agents[agent_handle])
    else:
        for agent in distance_map.agents:
            _path_for_agent(agent)

    return shortest_paths


class ShortestPathAfterSwitch(GraphPredictor):
    """
    Uses A* to construct shortest paths.
    For each agent if it is right before a switch and constructs the shortest paths 
        for each possible direction
    """
    IDX_DIST_TO_TARGET = 5
    IDX_DIST_TO_SWITCH = 4

    def __init__(self, max_depth: int = 20, 
            agent_priority_list: Optional[List[int]] = None, 
            selected_path_index: Optional[int] = 0,
            verify_if_cell_occupied: Optional[bool]=False):
        super().__init__(
            max_depth=max_depth, 
            agent_priority_list=agent_priority_list,
            verify_if_cell_occupied=verify_if_cell_occupied)
        self.selected_path_index = selected_path_index
    
    def reset(self, agent_priority_list: Optional[List[int]] = None):
        super().reset(
            agent_priority_list=agent_priority_list
        )
    
    def get_possible_paths_all_agents(self):
        """
        Return the shortest path for all agents
        """
        self.predicted_shortest_paths = get_shortest_path_from_switch(
            self.env.distance_map,
            max_depth=self.max_depth
        )
        for agent, paths in self.predicted_shortest_paths.items():
            tmp = (
                p
                for p in paths.values()
                if p is not None and all(adjacent(p[i-1], p[i]) for i in range(1, len(p)))
            )
            self.predicted_shortest_paths[agent] = dict(enumerate(tmp))
        return self.predicted_shortest_paths
    
    def get(self, handle: int = None):
        """
        Called whenever get_many in the observation build is called.
        Requires distance_map to extract the shortest path.
        
        If there is no shortest path, the agent just stands still and stops moving.

        Parameters
        ----------
        handle : int, optional
            Handle of the agent for which to compute the observation vector.

        Returns
        -------
        Dict[np.array], Dict[Dict[np.array]]
            Returns the shortest paths for each agents

        """
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
                                                3))
        self.occupied_at_timesteps.fill(np.nan)
        self.is_switch_grid = is_switch_for_grid(self.env.distance_map)
            
        possible_paths = {
            agent: {
                ind: self._get_agent_predicted_path(self.env.agents[agent], path)
                for ind, path in all_paths.items()
            }
            for agent, all_paths in self.get_possible_paths_all_agents().items()
        }

        # no prediction
        prediction = np.empty(shape=(self.max_depth + 1, 7))
        prediction[:, 0] = range(self.max_depth + 1)
        prediction[:, 1:] = -1

        prediction_dict = {}
        for agent_handle in range(number_of_agents):
            if self.selected_path_index in possible_paths[agent_handle]:
                prediction_dict[agent_handle] = possible_paths[agent_handle][self.selected_path_index]
            else:
                prediction_dict[agent_handle] = prediction

        return prediction_dict, possible_paths

    def _get_agent_predicted_path(self, agent, shortest_path):
        agent_handle = agent.handle
        # TODO: set element type for prediction to np.int
        prediction = np.empty(shape=(self.max_depth + 1, 7))

        if agent.position == agent.target:
            agent_virtual_position = agent.target
        else:
            agent_virtual_position = agent.position

        self.distance_map = self.env.distance_map.get()

        times_per_cell = 1
        agent_speed = 1
        offset = 0
        prediction[0] = [
            0, #timestep difference
            *agent_virtual_position, # position
            -1, # direction
            int(not self.is_switch_grid[agent_virtual_position[0], agent_virtual_position[1]]), # is position a simple cell (not a switch)
            agent_speed, # distance from the next positon -> (cummulative) distance from the target position
            0 # action to step on the current position
        ]

        # if there is a shortest path, remove the initial position
        if shortest_path:
            shortest_path = shortest_path[1:]
            # store the first timestep the current cell is occupied, the agent's new direction, the difference between 
            # the first and last timestep that the agent is occupying the current cell
            self.occupied_at_timesteps[agent_handle, agent_virtual_position[0], agent_virtual_position[1]] = [0, -1, 0]

        # initialize the current position
        new_position = agent_virtual_position
        action = 0 # no action

        for index in range(1, self.max_depth + 1):
            # if we're at the target, stop moving until max_depth is reached
            if new_position == agent.target or new_position[0] == -1:
                prediction[index] = [
                    index, 
                    -1, -1, 
                    -1, 
                    0, # agent is removed from the map => distance to next switch is set to 0
                    0, # distance from the target position
                    0 # no action needed
                ]
                continue
            # no path found
            if not shortest_path:
                prediction[index] = [
                    index, 
                    *new_position, 
                    -1, 
                    int(not self.is_switch_grid[agent_virtual_position[0], agent_virtual_position[1]]), # is position a simple cell (not a switch)
                    self.distance_map[
                        agent_handle,         
                        new_position[0],
                        new_position[1],
                    ], # distance from the target position
                    0 # no action
                ]
                continue

            if (index + offset) % times_per_cell == 0:
                # the agent arrived at the end of the current cell and enters the next grid cell
                # update current position
                action = _get_action_to_move(new_position, shortest_path[0])
                new_position = shortest_path[0]

                shortest_path = shortest_path[1:]

                # assume that the agent can enter the next position without causing a conflict 

                # label new_position as occupied at the current timestep 
                # save the agent direction and the difference between the first and last timestep when the agent
                #  is occupying the current position
                self.occupied_at_timesteps[agent_handle, new_position[0], new_position[1]] = [index + offset, -1, 0]
            else:
                # increase the length of when the current position is occupied
                self.occupied_at_timesteps[agent_handle, new_position[0], new_position[1], 2] += 1

            prediction[index] = [
                index, # difference between the timesteps
                *new_position, # position
                -1, # direction
                int(not( self.is_switch_grid[new_position[0], new_position[1]] or new_position == agent.target)), # is position a simple cell
                agent_speed, # dist from next position -> (cummulative) distance from the target position
                action #action performed to enter the cell
            ]

        # compute distance to next switch
        self._calc_next_switch_dist(prediction)
        dist_from_target = (
            self.distance_map[
                agent_handle,         
                int(prediction[-1][1]),
                int(prediction[-1][2]),
            ]
            if prediction[-1][1] >= 0 else 0
        )
        self._calc_dist_from_target(prediction, dist_from_target, agent_speed)
        
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
    
    def _calc_dist_from_target(self, prediction, dist_to_target, agent_speed):
        # compute distance to the target - dynamically from the route
        # distance fractions NOT allowed

        # if an agent arrived at the target once, it's not going to move
        # find the last index where dist_from_target > 0, previous values are also going to be > 0
        for _idx in range(self.max_depth, -1, -1):                                
            if prediction[_idx, self.IDX_DIST_TO_TARGET] > 0:
                prediction[_idx, self.IDX_DIST_TO_TARGET] = dist_to_target
                dist_to_target += agent_speed
        return prediction
