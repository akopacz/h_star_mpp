from exact_policy.two_agents_ts_priority import resolve_conflict_switch
from exact_policy.utils import STOP_MOVING, DO_NOT_WAIT
from utilities.conflicts import detect_common_switches
from votes.votes_utils import get_time_until_arrival, get_all_switches_from_obs,\
    get_actions_for_paths, normalize_votes, TARGET_NOT_FOUND_TSTEP, TARGET_NOT_FOUND_VOTE
import numpy as np


def evaluate_agent_path(handle, agent_path, obs, all_switches, speed, new_obs, all_actions_dict):
    if len(obs) == 1:
        # only one agent left on the map
        new_obs[handle][agent_path, 0] += 1
        
        return

    # every other agent
    for other_agent in obs:
        if other_agent == handle:
            continue
        # every other agent path
        for other_agent_path in obs[other_agent]:
            conflicts = detect_common_switches(
                agent1_switches=all_switches[handle][agent_path],
                agent2_switches=all_switches[other_agent][other_agent_path]
            )
            if conflicts:
                # at least one common position was found
                waiting_time = resolve_conflict_switch(
                    obs1=obs[handle][agent_path],
                    obs2=obs[other_agent][other_agent_path],
                    speed1=speed[handle],
                    speed2=speed[handle],
                    switches=conflicts,
                    prior=handle < other_agent
                )
                if waiting_time == DO_NOT_WAIT:
                    # the agent should move
                    # save the timestep, when the agent will reach its target
                    new_obs[handle][agent_path, 0] += 1
                elif waiting_time > 0:
                    # mark stop moving action
                    new_obs[handle][agent_path, 1] += 1

                # else -> we do not count it as a possible path -> do not vote
            else:
                new_obs[handle][agent_path, 0] += 1



def vote_based_on_conflicting_switches(obs, info, vote_size=None, agent=None):
    '''
    for each agent path determine the number of times when the agent should start moving 
    and cases when it should wait first

    Parameters
    ----------
    obs: Dict[int, Dict[ActionPathObservation object]]
        ActionPathObservation object observation for each agent.
    info : Dict
        additional information for each agent

    Returns
    -------
    transformed_obs: Dict
    '''
    if agent is None:
        agents = [a for a, o in obs.items() if o]  # filter empty observations
    else:
        agents = [agent] if agent in obs and obs[agent] else []
    if len(agents) == 0:
        return dict()
    if vote_size is None:
        vote_size = max(map(len, obs.values()))
    if len(agents) == 1:
        handle = agents[0]
        votes = np.zeros((vote_size, 3))
        for i in obs[handle]:
            votes[i, 0] = 1
            votes[i, 2] = obs[handle][i].dist_from_target[0]

        return {
            handle: votes
        }

    all_switches = get_all_switches_from_obs(obs)
    # get actions required to choose the current path
    all_actions_dict = get_actions_for_paths(obs)

    # actions advised by the exact algorithm
    transformed_obs = {
        handle: np.zeros((vote_size, 3))
        for handle in obs.keys()
    }
    for handle in agents:
        # at least 2 possible routes
        # generate every possible action -> evaluate the different agent paths
        # for every possible path
        for agent_path in obs[handle]:
            transformed_obs[handle][agent_path, 2] = obs[handle][agent_path].dist_from_target[0]
            if agent_path < vote_size:
                evaluate_agent_path(
                    handle,         # selected agent
                    agent_path,     # selected path for the current agent
                    obs,            # observations for all agents
                    all_switches,
                    info["speed"],
                    transformed_obs,       # all possible actions for agents -> is updated with current agent actions
                    all_actions_dict
                )

    return transformed_obs


def average_action_dict(action_dict, nr_of_agents, vote_size, actions_offset=0):
    """
    Calculate votes for different action based on the possible path lengths
    for each agent

    Parameters
    ----------
    action_dict: Dict[Tuple[int, int], int]
        Estimated time to reach target for possible agent-action pairs.
    nr_of_agents : int
        number of agents
    vote_size : int
        number of votes (output size for each agent)
    actions_offset : int, optional
        offset to shift with the actions to correspond indexes in the output vote vector
        if 0, the actions denote the indexes in the vote vector (for each agent)

    Returns
    -------
    _actions: Dict
        Dictionary with a vote vector of size vote_size,
        values in [0,1] interval or equal to TARGET_NOT_REACHED
        for each agent
    """

    # convert action dict to matrix format
    action_matrix = action_dict_to_matrix(
        action_dict, nr_of_agents, vote_size, actions_offset)

    action_matrix[:, :] = normalize_votes(
        action_matrix[:, :],
        masked_inds=(action_matrix[:, :] == TARGET_NOT_FOUND_VOTE))

    # return generated matrix as a dict
    return {
        # first n elements of the vote vector: votes for actions, next n elements: nr of conflicts
        agent: action_matrix[agent]
        for agent in range(nr_of_agents)
    }


def action_dict_to_matrix(action_dict, nr_of_agents, vote_size, actions_offset=0):
    """
    Convert the provided dictionary with agent actions to matrix format
    Shift each action if actions_offset is not 0
    """
    action_matrix = np.empty((nr_of_agents, vote_size))
    action_matrix[:, :] = TARGET_NOT_FOUND_VOTE

    # count votes generated by exact policy
    for agent, action in action_dict:
        if action_dict[agent, action] != TARGET_NOT_FOUND_TSTEP:
            # votes <- nr of timesteps the exact policy estimated reaching the target cell
            #    for the current action
            action_matrix[agent, action +
                          actions_offset] = action_dict[agent, action]
    return action_matrix
