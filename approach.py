# Author: Jonathan Roy
# Date: 29 March 2020

# This script implements the red light approach optimization


# IMPORTS
import numpy as np
from math import floor, ceil


# Approach Class implements the algorithm
class Approach:
    def __init__(self):
        # General params that are immutable
        self.v_min = 0 # obvious since cars don't back up
        self.x_max = 0 # value of x at stoplight

    # Set the initial conditions
    def setInitConditions(self, x_start):
        self.x_min = x_start # position at which vehicle starts at 

    # Set the parameters that characterize the world
    def setWorldParams(self, v_max, a_max):
        self.v_max = v_max
        self.a_max = a_max

    # Set the parameters used for computation
    def setComputeParams(self, x_step, v_step, t_step):
        self.x_step = x_step # discretization of position, in m
        self.v_step = v_step # discretization of velocity, in m/s
        self.t_step = t_step # length of time in seconds that a time step lasts

    def stateToIdx(self, state):

    def idxToState(self, idx):


    # This fn builds a state adjacency matrix
    # where the rows are states at t=k, and the columns are states at t=k+1
    # A True represents an edge
    def buildAdjancencyMatrix(self):
        self.state_size = # TODO SOME STATE SIZE FILL THIS IN -----------------------------------------------

        # Init boolean array
        self.adj_matrix = np.zeros((self.state_size, self.state_size), dtype=np.bool_) 

        # Iterate over all starting states
        for i in range(self.state_size):
            state = self.idxToState(i)
            v_min, v_max = state.v - self.a_max, state.v + self.a_max # min, max of reachable states
            v_min_discrete = ceil(v_min / self.v_step) * self.v_step
            v_max_discrete = floor(v_max / self.v_step) * self.v_step

            # Iterate over reachable velocities
            for v_new in np.arange(v_min_discrete, v_max_discrete, self.v_step): 
                v_avg = (state.v + v_new)/2 # avg velocity over timestep
                x_new = state.x + v_avg * self.t_step # change in position over timestep
                new_state = State(x_new, v_new) # TODO MAKE STATE CLASS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                self.adj_matrix[i, stateToIdx(new_state)] = True # set appropriate edge to True



    # This fn gives the value of a state at a point in time
    # Args:
    # state: a _____ containing the position and velocity
    # delta_t: the difference between t_eval and the current time
    # Returns: the value of a state if event G occurs at the passed time
    def rho(self, state, delta_t):
        x, v = state
        result = x + self.v_max * delta_t - 1/2 * 1/self.a_max * (self.v_max - v)**2

        # Check if you ran the red light
        if pos > 0: result = -999999999 # very bad value
        
        return result

    def findMaxNextState(self):

    def calcI(self):

    def forwardPass(self):
