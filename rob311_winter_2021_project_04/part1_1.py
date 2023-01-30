# part1_1.py: Project 4 Part 1 script
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
from mdp_cleaning_task import cleaning_env

## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the method  get_transition_model which creates the
    transition probability matrix for the cleanign robot problem desribed in the
    project document.
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""

def get_transition_model(env: cleaning_env) -> np.ndarray:
    """
    get_transition_model method creates a table of size (SxSxA) that represents the
    probability of the agent going from s1 to s2 while taking action a
    e.g. P[s1,s2,a] = 0.5
    This is the method that will be used by the cleaning environment (described in the
    project document) for populating its transition probability table

    Inputs
    --------------
        env: The cleaning environment

    Outputs
    --------------
        P: Matrix of size (SxSxA) specifying all of the transition probabilities.
    """

    P = np.zeros([len(env.states), len(env.states), len(env.actions)])

    ## START: Student Code

    # intend to go right
    # accidentally remains
    np.fill_diagonal(P[:, :, 1], 0.15)
    # accidentally goes left
    np.fill_diagonal(P[1:, :, 1], 0.05)
    #correctly goes right
    np.fill_diagonal(P[:, 1:, 1], 0.8)

    # intend to go left
    # accidentally remains
    np.fill_diagonal(P[:, :, 0], 0.15)
    # correctly goes left
    np.fill_diagonal(P[1:, :, 0], 0.8)
    # accidentally goes right
    np.fill_diagonal(P[:, 1:, 0], 0.05)

    # add terminal states
    P = np.array([P[i, :, :] if i not in env.terminal else np.zeros([len(env.states), len(env.actions)]) for i in env.states])

    ## END: Student code
    return P