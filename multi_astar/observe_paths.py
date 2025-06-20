from typing import Optional, List, Dict
import numpy as np
from collections import namedtuple


ActionPathObservation = namedtuple('ActionPathObservation',
                        'predicted_path '
                        'next_switch_dist '
                        'dist_from_target '
                        'position'
                        )

def equals(self, other):
    return (
        self.position == other.position 
    )
ActionPathObservation.__eq__ = equals

def coordinate_to_position(depth, coords):
    """
    Converts positions to coordinates::

         [ 0      d    ..  (w-1)*d
           1      d+1
           ...
           d-1    2d-1     w*d-1
         ]
         -->
         [ (0,0) (0,1) ..  (0,w-1)
           (1,0) (1,1)     (1,w-1)
           ...
           (d-1,0) (d-1,1)     (d-1,w-1)
          ]

    :param depth:
    :param coords:
    :return:
    """
    position = np.empty(len(coords), dtype=int)
    idx = 0
    for t in coords:
        # Set None type coordinates off the grid
        if t[0] == -1 or t[1] == -1:
            position[idx] = -1
        else:
            position[idx] = int(t[1] * depth + t[0])
        idx += 1
    return position

class ObserveManyPaths:
    def __init__(self, predictor):
        self.env = None
        self.predictor = predictor
        self.max_agent_path_length = predictor.max_depth

    def reset(self):
        pass

    def get_many(self, handles: Optional[List[int]] = None) -> Dict[int, Dict[int, ActionPathObservation]]:
        """
        We call the predictor ones for all the agents and use the predictions to generate our observations
        :param handles:
        :return:
        """

        _, self.all_predictions = self.predictor.get()

        if handles is None:
            handles = range(self.env.nr_agents)

        observations = {}
        for h in handles:
            observations[h] = self.get(h)

        return observations

    def get(self, handle: int = 0) -> Dict[int, ActionPathObservation]:
        """
        Custom observation.

        :param handle: index of an agent
        :return: Observation of handle
        """

        return {
            ind: ActionPathObservation(
                predicted_path=agent_prediction[:, 1:3], # list of agent's predicted positions(row, column, direction)
                next_switch_dist=agent_prediction[:, 4], # for each agent position: how many more steps needed untill the next switch
                dist_from_target=agent_prediction[:, 5], # for each agent position: how far is the target
                position=coordinate_to_position(self.env.width, agent_prediction[:, 1:3]),
            )
            for ind, agent_prediction in self.all_predictions[handle].items()
        }

    def set_env(self, env):
        self.env = env
        if self.predictor:
            self.predictor.set_env(self.env)