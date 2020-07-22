# Author: Jonathan Roy
# Date: 29 March 2020

# This script implements the red light approach optimization


# IMPORTS
import numpy as np
from math import floor, ceil

# This fn rounds things to the nearest discrete step
def round_to_step(value, step_size, behavior='round'):
    if behavior == 'round':
        result = round(value / step_size) * step_size 
    if behavior == 'floor':
        result = floor(value / step_size) * step_size 
    if behavior == 'ceil':
        result = ceil(value / step_size) * step_size 
    return result

# State class holds a vehicle state
class State:
    def __init__(self, pos, vel):
        self.v = vel
        self.x = pos


# Approach Class implements the algorithm
class Approach:
    def __init__(self):
        # General params that are immutable
        self.v_min = 0 # obvious since cars don't back up
        self.x_max = 0 # value of x at stoplight

    # Set the parameters used for computation
    def set_compute_params(self, x_step, v_step, t_step):
        self.x_step = x_step # discretization of position, in m
        self.v_step = v_step # discretization of velocity, in m/s
        self.t_step = t_step # length of time in seconds that a time step lasts

    # Set the parameters that characterize the world
    def set_world_params(self, v_max, a_max):

        # set v_max to the nearest smaller integer multiple of v_step
        self.v_max = floor(v_max / self.v_step) * self.v_step
        self.a_max = a_max

    # Set the initial conditions
    # x_start should be negative
    def set_init_conditions(self, x_start, v_start):
        assert(x_start < 0), 'x_start should be negative'
        self.x_min = round_to_step(x_start, self.x_step) # position at which vehicle starts at 
        v_start_discrete = round_to_step(v_start, self.v_step)
        self.initial_state = State(self.x_min, v_start_discrete)

    # Compute the size of the state space
    def compute_state_space_shape(self):
        num_v_steps = int((self.v_max - self.v_min) / self.v_step) + 1
        num_x_steps = int(abs(self.x_min - self.x_max) / self.x_step) + 1 
        self.state_space_shape = (num_x_steps, num_v_steps)
        self.state_space_flat_size = num_x_steps * num_x_steps

    # Convert state object to index
    def state_to_idx(self, state):

        # This equation gives a bijection from state to index
        idx = self.state_space_shape[1] * state.v + state.x
        return idx

    # Convert index to state object
    def idx_to_state(self, idx):
        x = idx % self.state_space_shape[1]
        v = idx // self.state_space_shape[1]
        return State(x, v)

    # This fn builds a state adjacency matrix
    # where the rows are states at t=k, and the columns are states at t=k+1
    # A True represents an edge
    def build_adjancency_matrix(self):

        # Init boolean array
        ss_flat_size = self.state_space_flat_size
        self.adj_matrix = np.zeros((ss_flat_size, ss_flat_size), dtype=np.bool_) 

        # Iterate over all starting states
        for i in range(ss_flat_size):
            state = self.idxToState(i)
            v_min, v_max = state.v - self.a_max, state.v + self.a_max # min, max of reachable states
            v_min_discrete = round_to_step(v_min, self.v_step, behavior='ceil')
            v_max_discrete = round_to_step(v_max, self.v_step, behavior='floor')

            # Iterate over reachable velocities
            for v_new in np.arange(v_min_discrete, v_max_discrete, self.v_step): 
                v_avg = (state.v + v_new)/2 # avg velocity over timestep
                x_new = state.x + v_avg * self.t_step # change in position over timestep
                x_new_discrete = round_to_step(x_new, self.x_step)
                new_state = State(x_new, v_new_discrete)
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

    # This fn take a state and a timestep, and returns the max value and location
    # of the states it can reach at the next timestep.
    def find_max_next_state(self, idx, delta_t):
        reachable_mask = self.adj_matrix[idx] # relevant row of adj_matrix
        next_state_vals = self.I[delta_t-1] # next timestep values of states
        reachable_vals = reachable_mask * next_state_vals 
        max_reachable = np.max(reachable_vals)
        argmax_reachable = np.argmax(reachable_vals)
        return (max_reachable, argmax_reachable) 

    def calc_I(self, idx, delta_t):
        # TODO make dist for event G
        alpha = # probabiliy of event G occurring right now
        max_reachable, argmax_reachable = self.find_max_next_state(idx, delta_t)

        # weighted value if light turns green right now
        cash_in_component = alpha * self.rho(idx, idx_to_state(idx), delta_t)

        # weighted expected value if light does not turn green right now
        not_cash_in_component = (1 - alpha) * max_reachable

        # add up for total expected value
        value = cash_in_component + not_cash_in_component

        # return expected value and pointer to next state
        return (value, argmax_reachable)

    # This runs the bulk of the algorithm. It calculates I recursively
    # for every state
    def backward_pass(self):
        pass
        
    # This fn can be run after I is defined everywhere from backward_pass().
    # it takes a path through the state space over time, and the result of 
    # this function is the desired behavior
    def forward_pass(self):
        pass