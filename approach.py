# Author: Jonathan Roy
# Date: 29 March 2020

# This script implements the red light approach optimization


# IMPORTS
from itertools import product
from math import floor, ceil
import numpy as np


# This fn rounds things to the nearest discrete step
# TODO: this could take a fn as argument instead of behavior str
def round_to_step(value, step_size, behavior='round'):
    if behavior == 'round':
        result = round(value / step_size) * step_size 
    if behavior == 'floor':
        result = floor(value / step_size) * step_size 
    if behavior == 'ceil':
        result = ceil(value / step_size) * step_size 
    return result

# Dist class holds a distribution representing the green light event
class Distribution:
    def __init__(self):
        self.dist = None 

    # Times are with reference to current time
    def uniform_dist(self, first_support, last_support, t_step):
        num_full_timesteps = floor((last_support - first_support)/t_step)
        num_timesteps = floor(last_support/t_step)
        self.dist = np.zeros(num_timesteps)
        support = np.ones(num_full_timesteps)/num_full_timesteps
        self.dist[-num_full_timesteps:] = support


# State class holds a vehicle state
class State:
    def __init__(self, pos, vel, bounds=None):
        self.x = pos
        self.v = vel
        if bounds:
            self.check_bounds(bounds)

    def check_bounds(self, bounds):
        if self.x < bounds[0] or \
           self.x > bounds[1] or \
           self.v < bounds[2] or \
           self.v > bounds[3]:
            raise IndexError('State does not lie within state space')

    def __repr__(self):
        return f'State object: x = {self.x}; v = {self.v}'

    def __eq__(self, s):
        if isinstance(s, State):
            return s.x == self.x and s.v == self.v
        else:
            return False


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
        self.v_max = round_to_step(v_max, self.v_step, behavior='floor')
        self.a_max = a_max

        # set traffic light distribution
    def set_traffic_light_params(self, first_support, last_support):
        self.green_dist = Distribution()
        self.green_dist.uniform_dist(first_support, last_support, self.t_step)
        self.calc_t_eval(last_support)

    # calc_t_eval() calculates t_eval, which is the time at which the vehicle will be back up to
    # speed in the worst case.
    def calc_t_eval(self, last_support):
        # eval_gap is the time it takes to get up to top speed in worst case
        eval_gap = self.v_max / self.a_max
        self.t_eval = last_support + eval_gap

    # Set the initial conditions
    # x_start should be negative
    def set_init_conditions(self, x_start, v_start):
        assert(x_start < 0), 'x_start should be negative'
        self.x_min = round_to_step(x_start, self.x_step) # position at which vehicle starts at 
        v_start_discrete = round_to_step(v_start, self.v_step)
        self.initial_state = State(self.x_min, v_start_discrete)

        # Set time to zero
        # This might be unnecessary, since we don't implement algorithm results here
        self.t = 0

    # Compute the size of the state space
    def compute_state_space_shape(self):
        num_v_steps = int((self.v_max - self.v_min) / self.v_step) + 1
        # Use abs() because x is negative
        num_x_steps = int(abs(self.x_max - self.x_min) / self.x_step) + 1
        self.state_space_shape = (num_x_steps, num_v_steps)
        self.state_space_bounds = [self.x_min, self.x_max, self.v_min, self.v_max]

    # Discretize state to position and velocity steps
    def discretize_state(self, state):
        new_x = round_to_step(state.x, self.x_step)
        new_v = round_to_step(state.v, self.v_step)
        
        # Use a new State object so that boundaries get checked
        return State(new_x, new_v)

    # Convert state to indices in a state space shaped matrix
    def state_to_indices(self, state):
        state = self.discretize_state(state)
        return (int(state.x / self.x_step * -1), int(state.v / self.v_step))

    # Convert indices(x,v) to state
    def indices_to_state(self, indices):
        return State(indices[0] * self.x_step * -1, indices[1] * self.v_step, self.state_space_bounds)
      
        

    # This fn builds a state adjacency matrix
    # where the rows are states at t=k, and the columns are states at t=k+1
    # A True represents an edge
    # Args:
    #     none
    # Returns:
    #     none
    def build_adjacency_matrix(self):

        # Init boolean array
        ss_size = self.state_space_shape
        self.adj_matrix = np.zeros((ss_size * 2), dtype=np.bool_) 

        # Iterate over all starting states
        for i, j in product(range(ss_size[0]), range(ss_size[1])):
            state = self.indices_to_state((i, j))
            a_increment = self.a_max * self.t_step # max acceleration that can occur in a timestep
            v_min, v_max = state.v - a_increment, state.v + a_increment # min, max of reachable states
            v_min_discrete = round_to_step(v_min, self.v_step, behavior='ceil')
            v_max_discrete = round_to_step(v_max, self.v_step, behavior='floor')

            # Iterate over reachable velocities
            for v_new in np.arange(v_min_discrete, v_max_discrete, self.v_step): 

                # Compute new position 
                # (this linear approx is okay because it gets better as timestep gets smaller)
                v_avg = (state.v + v_new)/2 # avg velocity over timestep
                x_new = state.x + v_avg * self.t_step # change in position over timestep
                x_new_discrete = round_to_step(x_new, self.x_step)

                # Set relevant element of adjacency matrix to True
                try:
                    new_state = State(x_new_discrete, v_new, self.state_space_bounds)
                except IndexError: # If the state is out of bounds
                    pass
                else:
                    i_new, j_new = self.state_to_indices(new_state)
                    self.adj_matrix[i, j, i_new, j_new] = True # set appropriate edge to True

    # rho() gives the value of a state at a point in time, if the event G has already occurred.
    # It reflects the position at t_eval if the vehicle accelerates at a_max until reaching v_max.
    # Args:
    #     state: a State object containing the position and velocity
    #     delta_t: the difference between t_eval and the current time
    # Returns: 
    #     the value of a state if event G occurs at the passed time
    def rho(self, state, delta_t):
        x, v = state.x, state.v

        # Derivation shown in notes from 4/3/2020
        result = x + self.v_max * delta_t - 1/2 * 1/self.a_max * (self.v_max - v)**2

        # Check if you ran the red light
        if pos > 0: result = -999999999 # very bad value
        
        return result

    # find_max_next_state() takes a state and a timestep, and returns the max value and location
    # of the states it can reach at the next timestep.
    # Args:
    #     idx: index of state from which to look
    #     timestep: algorithm timestep from which to look 
    # Returns: 
    #     tuple: (value of max reachable state, index of max reachable state)
    def find_max_next_state(self, idx, timestep):
        reachable_mask = self.adj_matrix[idx] # relevant row of adj_matrix
        next_state_vals = self.I[timestep-1] # next timestep values of states
        reachable_vals = reachable_mask * next_state_vals 
        max_reachable = np.max(reachable_vals)
        argmax_reachable = np.argmax(reachable_vals)
        return (max_reachable, argmax_reachable) 

    #TODO double check time variables 
    # delta_t is t_eval-t_current! Or the amount of time until t_eval.
    def calc_I(self, idx, timestep):

        # alpha is the probability of event G occurring right now
        alpha = self.green_dist.dist[timestep]
        max_reachable, argmax_reachable = self.find_max_next_state(idx, timestep)

        # weighted value if light turns green right now
        cash_in_component = alpha * self.rho(self.idx_to_state(idx), timestep)

        # weighted expected value if light does not turn green right now
        not_cash_in_component = (1 - alpha) * max_reachable

        # add up for total expected value
        value = cash_in_component + not_cash_in_component

        # return expected value and pointer to next state
        return (value, argmax_reachable)

    # This runs the bulk of the algorithm. It calculates I recursively
    # for every state.
    # It must proceed backwards in time because the earlier states' values
    # depend on the later states' values
    def backward_pass(self):

        # calculate number of timesteps
        num_timesteps = len(self.green_dist.dist) # This should work out to be the number of timesteps from 0 to last_support

        # Initialize I to timesteps x statespace size
        self.I = np.zeros((num_timesteps, self.state_space_flat_size))

        # Iterate through all timesteps to fill out I
        # This progresses backwards in clock time through I, which has rows in reverse chronological order
        for I_row, timestep in enumerate(range(num_timesteps - 1, -1, -1)):

            # Iterate through all states 
            for idx in range(self.state_space_flat_size):
                self.I[I_row, idx] = self.calc_I(idx, timestep)
                 
             


    # This fn can be run after I is defined everywhere from backward_pass().
    # it takes a path through the state space over time, and the result of 
    # this function is the desired behavior
    def forward_pass(self):
        pass
