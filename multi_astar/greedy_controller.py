import random
from typing import Dict

import numpy as np
from votes.full_paths import vote_based_on_conflicting_switches
from votes.votes_utils import _get_action_to_move
from multi_astar.controller import KPathController

class GreedyController(KPathController):
    """
    for each agent observation (+ info ex. speed) -> action
    """

    def __init__(self, preprocess_obs, nr_of_actions, vote_size=None,
                 max_agent_path_length=None, actions_offset=0) -> None:
        super().__init__(None, preprocess_obs, nr_of_actions, None, 
                         vote_size, max_agent_path_length, actions_offset)

    def plan_action_combined_policy(self, obs, info, preprocess_obs=True, **_):
        action_dict = self.get_shortest_path_actions(obs, info)
        transformed_obs = None

        if preprocess_obs:
            # preprocess observation
            # convert agent observations to numpy matrixes
            transformed_obs = self.get_processed_obs(obs, info)

            if len(transformed_obs) > 1 and sum(info['action_required'].values()) > 1:
                # more than 1 agent present 
                # use complex policy to determine actions
                acts = self.act_tr_policy(
                    obs=transformed_obs,
                    action_required=info['action_required'],
                )
                action_dict.update({
                    agent: _get_action_to_move(obs[agent][ind].predicted_path[0], obs[agent][ind].predicted_path[1])
                    for agent, ind in acts.items()
                    if ind in obs[agent]
                })

        # remove unnecessary actions
        for agent, act_req in info['action_required'].items():
            if not act_req and agent in action_dict:
                del action_dict[agent]

        return action_dict, transformed_obs
    def get_processed_obs(self, obs, info):
        return vote_based_on_conflicting_switches(obs, info, self.vote_size)
    
    def save_tr_policy(self, filename, episode=None, score=None):
        return

    def load_tr_policy(self, *_):
        return

    def act_tr_policy(self, obs, action_required, preprocessed=False, **_) -> Dict[int, int]:
        if preprocessed:
            _obs = obs
        else:
            _obs = self._preprocess_obs_many(obs)

        return {
            agent: self.optimal(_obs[agent]) - self.actions_offset
            for agent, move in action_required.items()
            if move
        }

    def optimal(self, array):
        m = array[:, 0].max()
        IND = 2
        column = min(r[IND] for r in array if r[0] == m)
        return next(i for i in range(array.shape[0]) if array[i,0] == m and array[i,IND]==column)
            