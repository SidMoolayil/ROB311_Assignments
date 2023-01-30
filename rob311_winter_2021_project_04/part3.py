
# part3_solution.py  (adopted from the work of Anson Wong)
#
# --
# Artificial Intelligence
# ROB 311 Winter 2021
# Programming Project 4
#
# --
# University of Toronto Institute for Aerospace Studies
# Stars Lab
#
# Course Instructor:
# Matthew Giamou
# mathhew.giamau@robotics.utias.utoronto.ca
#
# Teaching Assistant:
# Sepehr Samavi
# sepehr@robotics.utias.utoronto.ca
#
# Abhinav Grover
# abhinav.grover@robotics.utias.utoronto.ca

"""
 We set up bandit arms with fixed probability distribution of success,
 and receive stochastic rewards from each arm of +1 for success,
 and 0 reward for failure.
"""
import numpy as np

class MAB_agent:
    """
        TODO:
        Implement the get_action and update_state function of an agent such that it 
        is able to maximize the reward on the Multi-Armed Bandit (MAB) environment.
    """
    def __init__(self, num_arms=5):
        self.__num_arms = num_arms #private
        ## IMPLEMENTATION
        # initialization of t, action rewards, and number of each actions
        self.t = 0
        self.reward = [0 for i in range(num_arms)]
        self.n = [1 for i in range(num_arms)]

    def update_state(self, action, reward):
        """
            TODO:
            Based on your choice of algorithm, use the the current action and 
            reward to update the state of the agent. 
            Optinal function, only use if needed.
        """
        ## IMPLEMENTATION
        # incrementally add reward(a) and update N(a)
        self.reward[action] += reward
        self.n[action] += 1
        pass

    def get_action(self) -> int:
        """
            TODO:
            Based on your choice of algorithm, generate the next action based on
            the current state of your agent.
            Return the index of the arm picked by the policy.
        """
        ## IMPLEMENTATION
        # increment t
        self.t += 1

        # intervals of exploration tuned empirically
        if self.t < 12:
            return np.random.randint(0,self.__num_arms)
        elif self.t < 38 and self.t > 25:
            return np.random.randint(0,self.__num_arms)
        elif self.t < 70 and self.t > 60:
            return np.random.randint(0,self.__num_arms)
        elif self.t < 90 and self.t > 75:
            return np.random.randint(0,self.__num_arms)
        elif self.t < 150 and self.t > 100:
            return np.random.randint(0,self.__num_arms)

        # estimated (average) rewards for all actions
        Q = [self.reward[arm]/self.n[arm] for arm in range(self.__num_arms)]

        # tested with uncertainty parameter, ultimately not used
        #uncertainty = [2 * np.sqrt(np.log(self.t)/self.n[arm]) for arm in range(self.__num_arms)]
        #if self.t > 0:
        #    Q = np.add(Q,uncertainty)

        # return action with maximum expected reward
        return np.argmax(Q)

        raise NotImplementedError