# Author: Jonathan Roy
# Date: 29 March 2020

# This script implements the red light approach optimization


# IMPORTS
from itertools import product
import json
from math import floor, ceil
import numpy as np

from progress.bar import IncrementalBar

from distribution import Distribution
from state import State
from utils import round_to_step


# Approach Class implements the algorithm
class Approach:

    def __init__(self, json_path=None):
        # General params that are immutable
        self.v_min = 0 # obvious since cars don't back up
        self.x_max = 0 # value of x at stoplight

        if json_path:
            self.configure(json_path)

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

    # this fn takes a json file and configures all the stuff
    # Includes parameters, initial conditions, and trivial consequences of these
    def configure(self, json_path):
        with open(json_path) as f:
             test1 = json.load(f)
        self.set_compute_params(x_step=test1['compute_params']['x_step'],
                                v_step=test1['compute_params']['v_step'],
                                t_step=test1['compute_params']['t_step'])
        self.set_world_params(v_max=test1['world_params']['v_max'],
                              a_max=test1['world_params']['a_max'])
        self.set_traffic_light_params(first_support=test1['traffic_light_params']['first_support'],
                                      last_support=test1['traffic_light_params']['last_support'])
        self.set_init_conditions(x_start=test1['init_conditions']['x_start'],
                                 v_start=test1['init_conditions']['v_start'])
        self.compute_state_space_shape()

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
        
    # Calculate new vehicle position
    # mode refers to Riemann integration mode
    def delta_x(self, state, v_new, mode='trapezoidal'):
        if mode == 'trapezoidal':
           v_avg = (state.v + v_new)/2
           x_new = state.x + v_avg * self.t_step
           x_new_discrete = round_to_step(x_new, self.x_step)
        if mode == 'right':
           x_new = state.x + v_new * self.t_step
           x_new_discrete = round_to_step(x_new, self.x_step)
        return x_new_discrete

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
                x_new_discrete = self.delta_x(state, v_new, mode='trapezoidal')

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
    #     timestep: 
    # Returns: 
    #     the value of a state if event G occurs at the passed time
    # TODO timestep does not map to seconds. Fix this with indx to clock time conversions like state space variables
    def rho(self, state, timestep):
        x, v = state.x, state.v
        # delta_t: the difference between t_eval and the current time
        delta_t = self.t_eval - timestep 

        # Derivation shown in notes from 4/3/2020
        result = x + self.v_max * delta_t - 1/2 * 1/self.a_max * (self.v_max - v)**2

        # Check if you will run the red light at next timestep
        v_next_min = max((state.v - self.a_max * self.t_step), self.v_min) # maxed with v_min to avoid negative velocity
        x_next_min = x + self.delta_x(state, v_next_min, mode='trapezoidal')
        if x_next_min > 0: result = -999999999 # very bad value
        
        return result

    # find_max_next_state() takes a state and a timestep, and returns the max value and location
    # of the states it can reach at the next timestep.
    # Args:
    #     idx: index of state from which to look
    #     timestep: algorithm timestep from which to look 
    # Returns: 
    #     tuple: (value of max reachable state, index of max reachable state)
    def find_max_next_state(self, state_indices, timestep):
        reachable_mask = self.adj_matrix[state_indices[0], state_indices[1]] # relevant slice of adj_matrix
        next_state_vals = np.zeros(self.state_space_shape)
        for i, j in product(range(self.state_space_shape[0]), range(self.state_space_shape[1])):
            next_state_vals[i, j] = self.I[timestep+1, i, j, 0]
        reachable_vals = reachable_mask * next_state_vals 
        max_reachable = np.max(reachable_vals)
        argmax_reachable = np.argmax(reachable_vals) #This stores the index raveled. Use np.unravel_index() to rectify
        return (max_reachable, argmax_reachable) 

    def calc_I(self, state_indices, timestep):

        # alpha is the probability of event G occurring right now
        alpha = self.green_dist.dist[timestep]
        if timestep != self.num_timesteps -1:
            max_reachable, argmax_reachable = self.find_max_next_state(state_indices, timestep)
        else:
            max_reachable = 0
            argmax_reachable = 0 # This should never get used in the forward pass

        # weighted value if light turns green right now
        cash_in_component = alpha * self.rho(self.indices_to_state(state_indices), timestep)

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

        ss_shape = self.state_space_shape

        # calculate number of timesteps
        self.num_timesteps = len(self.green_dist.dist) # This should work out to be the number of timesteps from 0 to last_support

        # Initialize I to timesteps x statespace size
        # Last dimension stores I and pointer to next state: [I, pointer_index]
        self.I = np.zeros((self.num_timesteps, ss_shape[0], ss_shape[1], 2))

        # Iterate through all timesteps to fill out I
        # This progresses backwards in clock time through I, which has rows in reverse chronological order
        bar = IncrementalBar('Calculating I', max=self.num_timesteps)
        for timestep in range(self.num_timesteps - 1, -1, -1):

            # Iterate through all states 
            for i, j in product(range(ss_shape[0]), range(ss_shape[1])):
                self.I[timestep, i, j] = self.calc_I((i, j), timestep)

            bar.next()
        bar.finish()
                 
    # This fn takes one step in time forward through I
    def forward_step(self, state, timestep):
        
        # Initial state indices
        i, j = self.state_to_indices(state)

        # Iterate forward through timesteps
        next_indices_raveled = int(self.I[timestep, i, j, 1])
        next_state_indices = np.unravel_index(next_indices_raveled, self.state_space_shape) 
        next_state = self.indices_to_state(next_state_indices)

        return (next_state, next_state_indices)

    # This fn can be run after I is defined everywhere from backward_pass().
    # it takes a path through the state space over time, and the result of 
    # this function is the desired behavior
    def forward_pass(self):

        # Initialize list to store path through state space
        state_list = [self.initial_state]

        # Iterate forward through timesteps
        current_state = self.initial_state
        for timestep in range(1, self.num_timesteps):
            next_state, _ = self.forward_step(current_state, timestep)
            current_state = next_state
            state_list.append(current_state)

        return state_list

















 
