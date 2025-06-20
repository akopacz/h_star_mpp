from typing import List, NamedTuple, Tuple

ACT_DICT = {
    0: (-1, 0),  # up
    1: (0, -1),  # left
    2: (1, 0),  # down
    3: (0, 1),  # right
    4: (0, 0)  # stopping/halt
}
ACTIONS = tuple(ACT_DICT.keys())

reverse_dict = {
    (-1, 0): 0,  # up
    (0, -1): 1,  # left
    (1, 0): 2,  # down
    (0, 1): 3,  # right
    (0, 0): 4,  # stop
}

NEIGHBORS = ((0, -1), (0, 1), (-1, 0), (1, 0))

Waypoint = NamedTuple("Waypoint", [
    ("position", Tuple[int, int]),
])

NextAction = NamedTuple('NextAction',
                        [('action', int),
                         ('next_position', Tuple[int, int]),
                         ])


class Agent:
    handle = None
    position = (-1, -1)
    initial_position = (-1, -1)
    prev_position = (-1, -1)
    target = (-1, -1)
    finished = False

    def __init__(self, handle, initial_position, target) -> None:
        self.handle = handle
        self.initial_position = tuple(initial_position)
        self.target = tuple(target)
        self.reset()

    def reset(self) -> None:
        self.position = self.initial_position
        self.prev_position = self.initial_position
        self.finished = False

    def update_position(self, new_pos):
        self.prev_position = self.position
        self.position = new_pos
        if self.position == self.target:
            self.finished = True


class Env:
    agents: List[Agent] = None
    maze = None
    distance_map = None
    obs_builder = None
    nr_agents: int = None
    height: int = None
    width: int = None
    collision_penalty: int = -1
    not_finished_penalty: int = 0
    finished_reward: int = 1
    max_step: int = -1

    def __init__(self, maze, agents, obs_builder, distance_map, collision_penalty=None) -> None:
        self.maze = maze
        self.height, self.width = self.maze.shape
        self.agents = agents
        self.nr_agents = len(agents)
        self.max_steps = max((self.height + self.width) * 2 * self.nr_agents, 100)
        self.distance_map = distance_map
        self.distance_map.calc_dist_map(self.maze)
        self.obs_builder = obs_builder
        self.obs_builder.set_env(self)
        if collision_penalty is not None:
            self.collision_penalty = collision_penalty
        self.reset()

    def reset(self):
        for ag in self.agents:
            ag.reset()
        self.distance_map.reset()

    def step(self, action_dict):
        collides = {a: False for a in range(self.nr_agents)}
        # step
        for agent in self.agents:
            handle = agent.handle
            act = action_dict.get(handle)
            if agent.finished or act is None:
                # agent.update_position(agent.position)
                continue
            
            new_pos = (agent.position[0] + ACT_DICT[act]
                       [0], agent.position[1] + ACT_DICT[act][1])
            # step
            if new_pos[0] >= 0 and new_pos[1] >= 0 and new_pos[0] < self.height and new_pos[1] < self.width and self.maze[new_pos] == 0:

                # check if step is ok
                for agent2 in self.agents:
                    if agent2.handle != handle and agent2.position == new_pos:
                        # collision detected
                        # undo step
                        collides[handle] = True
                        break
            else:
                collides[handle] = True
            if not collides[handle]:
                # do distance map update before moving the agent, 
                # so agent.position is still the old position
                self.distance_map.update_blocked_cell(agent.handle, new_pos, agent.position)
                agent.update_position(new_pos)
            else:
                agent.update_position(agent.position)
        for agent in self.agents:
            blocked = next(cell for cell, ag in self.distance_map.blocked_cells.items() if agent.handle in ag)
            assert(blocked == self.distance_map.lookup[agent.position])
        # reward
        reward_dict = {}
        for agent in self.agents:
            if agent.finished:
                reward_dict[agent.handle] = self.finished_reward
            elif collides[agent.handle]:
                reward_dict[agent.handle] = self.collision_penalty
            else:
                reward_dict[agent.handle] = self.not_finished_penalty
        # observations
        obs_dict = self.obs_builder.get_many()

        return obs_dict, reward_dict


def _get_action_to_move(coords1, coords2):
    # default action is to wait
    return reverse_dict.get((coords2[0]-coords1[0], coords2[1]-coords1[1]), 4)