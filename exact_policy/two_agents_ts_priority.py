from exact_policy.utils import DO_NOT_WAIT,\
    DEFAULT_WAIT, TARGET_NOT_FOUND_TSTEP 


def resolve_conflict_switch(obs1, obs2, speed1, speed2, switches, prior=True):
    '''
    Determines an action for the first agent

    Parameters
    ----------
    obs1: object
        observation for agent 1
    obs2: object
        observation for agent 2
    speed1: float
        agent 1 speed
    speed2: float
        agent 1 speed
    switches: Dict[int,ConflictingSwitch]
        the common positions of agent 1 with agent 2
    prior: bool
        agent 1 has priority when choosing action over agent 2 

    Returns
    -------
    int: timestep offset for agent 1 to perform the first move action
    '''
    def _switch_passed(cell_steps, next_switch_dist, position):
        agent_pos = position[0]
        next_ind = 1
        while next_ind < cell_steps and next_switch_dist[next_ind] == agent_pos:
            next_ind += 1

        if next_ind < cell_steps:
            # the agent already passed the position where it can change its action
            agent_next_pos = position[next_ind]
            return agent_pos in switches or agent_next_pos in switches
        else:
            return agent_pos in switches

    cell_steps1 = int(1 / speed1)
    cell_steps2 = int(1 / speed2)

    # check if decomposition is needed
    #  <=> if the agents passed the last position where they can make a decision
    agent1_passed_switch = _switch_passed(
        cell_steps1, obs1.next_switch_dist, obs1.position)
    agent2_passed_switch = _switch_passed(
        cell_steps2, obs2.next_switch_dist, obs2.position)
    if agent1_passed_switch and agent2_passed_switch:
        # deadlock situation
        return TARGET_NOT_FOUND_TSTEP
    if agent1_passed_switch and not agent2_passed_switch:
        # the first agent passed it's conflicting switch => must move forward
        return DO_NOT_WAIT
    if agent2_passed_switch:
        # the second switch is going to be after the 2nd agent's current position
        # # 2nd agent is in the critical section -> stop
        return DEFAULT_WAIT

    if prior:
        waiting_time = DO_NOT_WAIT
    else:
        waiting_time = DEFAULT_WAIT

    return waiting_time
