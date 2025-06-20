from typing import Dict

import numpy as np
from votes.full_paths import vote_based_on_conflicting_switches
from votes.votes_utils import _get_action_to_move


class KPathController:
    """
    for each agent observation (+ info ex. speed) -> action
    """

    def __init__(self, train_policy, preprocess_obs, nr_of_actions, is_data_valid=None, vote_size=None,
                 max_agent_path_length=None, actions_offset=0) -> None:
        self.training_policy = train_policy
        self.preprocess = preprocess_obs
        self.is_data_valid = lambda x: True if is_data_valid is None else is_data_valid
        self.prev_obs = None
        self.vote_size = vote_size if vote_size is not None else nr_of_actions
        self.nr_of_actions = nr_of_actions
        self.max_agent_path_length = max_agent_path_length
        self.actions_offset = actions_offset

    def reset(self, nr_of_agents, env_width, initial_obs=None) -> None:
        self.nr_of_agents = nr_of_agents
        self.env_width = env_width
        if initial_obs is not None:
            self.prev_obs = self._preprocess_obs_many(initial_obs)
        else:
            self.prev_obs = None

    def _preprocess_obs_many(self, obs) -> Dict[int, np.ndarray]:
        return {
            agent: self.preprocess(obs[agent])
            for agent in obs
        }

    def get_first_available_actions(self, obs, info):
        return self.get_action_dict(
            observations={
                agent: agent_obs[0]
                for agent, agent_obs in obs.items()
                if 0 in agent_obs
            },
            info=info
        )

    def get_shortest_path_actions(self, obs, info):
        # baseline -- follow shortest paths
        return self.get_action_dict(
            observations={
                agent: agent_obs[min(
                    agent_obs, key=lambda k: agent_obs[k].dist_from_target[0])]
                for agent, agent_obs in obs.items()
                if 0 in agent_obs
            },
            info=info
        )

    def get_processed_obs(self, obs, info):
        return {
            ag: o.reshape(-1, 1)
            for ag, o in vote_based_on_conflicting_switches(obs, info, self.vote_size).items()
        }

    def train_rl_policy(self, actions, rewards, obs, done, action_required) -> None:
        if self.training_policy is None or self.prev_obs is None:
            self.prev_obs = self._preprocess_obs_many(obs)
            return
        next_obs = self._preprocess_obs_many(obs)
        for agent in action_required:
            if action_required[agent] and agent in actions and self.is_data_valid(self.prev_obs[agent]):
                self.training_policy.step(
                    state=self.prev_obs[agent],
                    action=actions[agent] + self.actions_offset,
                    reward=rewards[agent],
                    next_state=next_obs[agent],
                    done=done[agent]
                )
        self.prev_obs = next_obs.copy()

    def save_tr_policy(self, filename, episode=None, score=None):
        self.training_policy.save(filename, episode, score)

    def load_tr_policy(self, filename):
        return self.training_policy.load(filename)

    def get_action_dict(self, observations, info):
        """ Returns action dict based on the actions given by the observations"""
        action_dict = {}
        for handle, obs in observations.items():
            # check if action is required
            if info['action_required'][handle]:
                action_dict[handle] = _get_action_to_move(
                    obs.predicted_path[0], obs.predicted_path[1])
        return action_dict

    def act_tr_policy(self, obs, action_required, eps=0., preprocessed=False) -> Dict[int, int]:
        if preprocessed:
            _obs = obs
        else:
            _obs = self._preprocess_obs_many(obs)
        return {
            agent: self.training_policy.act(_obs[agent], eps=eps)
            for agent, move in action_required.items()
            if move
        }
