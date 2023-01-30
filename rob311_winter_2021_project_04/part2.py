# part2.py: Project 4 Part 2 script
#
# --
# Artificial Intelligence
# ROB 311 Winter 2020
# Programming Project 4
#
# --
# University of Toronto Institute for Aerospace Studies
# Stars Lab
#
# Course Instructor:
# Dr. Jonathan Kelly
# jkelly@utias.utoronto.ca
#
# Teaching Assistant:
# Matthew Giamou
# mathhew.giamau@robotics.utias.utoronto.ca
#
# Abhinav Grover
# abhinav.grover@robotics.utias.utoronto.ca

###
# Imports
###

import numpy as np
from mdp_env import mdp_env
from mdp_agent import mdp_agent


## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the policy_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""


def policy_iteration(env: mdp_env, agent: mdp_agent, max_iter = 1000) -> np.ndarray:
    """
    policy_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (4ed pg 657). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs-
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        max_iter: Max iterations for the algorithm

    Outputs -
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
       <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    np.random.seed(1) # TODO: Remove this

    policy = np.random.randint(len(env.actions), size=(len(env.states), 1))
    agent.utility = np.zeros([len(env.states), 1])

    ## START: Student code
    # initialization
    iterCount = 0
    done = False

    # loop until policy improvement step yields no change in the utilities, or max iterations reached
    while not done and iterCount < max_iter:
        # U <- Policy Evaluation
        agent.utility = [agent.gamma*np.sum([env.transition_model[i, j, policy[i]] * (agent.utility[j]) for j in env.states])+env.rewards[i] for i in env.states]
        # assume utilities will be unchanged
        done = True
        # looping over states
        for s in env.states:
            # if policy better than current policy
            if max([np.sum([env.transition_model[s, j, k] * (agent.utility[j]) for j in env.states]) for k in env.actions]) > np.sum([env.transition_model[s, j, policy[s]] * (agent.utility[j]) for j in env.states]):
                # overwrite policy
                policy[s] = np.argmax([np.sum([env.transition_model[s, j, k] * (agent.utility[j]) for j in env.states]) for k in env.actions])
                # record that utilities were changed
                done = False
        # increment count
        iterCount += 1
    policy=np.squeeze(policy)
    ## END: Student code

    return policy