import numpy as np
from typing import Dict, NamedTuple

ConflictingSwitch = NamedTuple('ConflictingSwitch', [
    ('tstep1', int),
    ('tstep2', int),
])


def detect_common_switches(agent1_switches, agent2_switches):
    """
    Returns a set of possible conflicting switches

    Parameters
    ----------
    agent1_switches: [Dict[int]
        a dictionary with keys as the switch positions 
        and values as the timesteps when the first agent is at the key position
    agent2_switches: [Dict[int]
        a dictionary with keys as the switch positions 
        and values as the timesteps when the second agent is at the key position

    Returns
    -------
    Dict[ConflictingSwitch]: the common positions from agent1_switches and agent2_switches
    """
    return {
        c_switch: ConflictingSwitch(
            tstep1=agent1_switches[c_switch],
            tstep2=agent2_switches[c_switch]
        )
        for c_switch in agent1_switches.keys() & agent2_switches.keys()
        if c_switch != -1
    }

def detect_common_switches_many(all_switches, handle):
    """
    Returns a set of possible conflicting switches

    Parameters
    ----------
    all_switches: Dict[Dict[int]] 
        stores for each agent 
        a dictionary with keys as the switch positions 
        and values as the timesteps when the current agent is at the key position
    handle: int
        agent handle to check the conflicts for

    Returns
    -------
    Dict[Dict[ConflictingSwitch]]: the common positions of other agents and the current agent
    """
    return {
        other_agent: {
            c_switch: ConflictingSwitch(
                tstep1=all_switches[handle][c_switch],
                tstep2=all_switches[other_agent][c_switch]
            )
            for c_switch in all_switches[handle].keys() & all_switches[other_agent].keys()
            if c_switch != -1
        }
        for other_agent in all_switches.keys()
        if handle != other_agent
    }