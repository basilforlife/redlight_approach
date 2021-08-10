import json
from itertools import product
from typing import Any, List, Optional, Tuple

import numpy as np
from progress.bar import IncrementalBar

from redlight_approach.approach_utils import round_to_step, timer
from redlight_approach.distribution import ArbitraryDistribution, UniformDistribution
from redlight_approach.state import State


# Approach Class implements the algorithm
class Approach:
    """Represents a red light approach scenario and computes a time optimal approach.

    This class builds a representation of a scenario where a vehicle approaches a
    signalized intersection while the light is red. It computes the value of every reachable
    state at every discrete timestep after the initialization time, over a discretized (position, velocity)
    state space. The value function values states according to the expectation of time spent travelling.
    It computes the optimal motion plan through the discrete state space over time.
    """

    def __init__(self, json_path: Optional[str] = None) -> None:
        """Initialize Approach object, and optionally configure from a file

        Parameters
        ----------
        json_path
            The file path where the configuration file exists
        """
        self.v_min = 0  # obvious since cars don't back up
        self.x_max = 0  # value of x at stoplight

        if json_path:
            self.configure(json_path)
        else:
            self.params = {}

    def __repr__(self) -> str:
        return f"Approach object: configured with {self.params}"

    def __eq__(self, other: Any) -> bool:
        raise NotImplementedError

    def set_compute_params(self, x_step: float, v_step: float, t_step: float) -> None:
        """Sets the size of discrete steps for the state space and time

        Parameters
        ----------
        x_step
            Discrete step size of position dimension [m]
        v_step
            Discrete step size of velocity dimension [m/s]
        t_step
            Length of one timestep [s]
        """
        self.x_step = x_step
        self.v_step = v_step
        self.t_step = t_step

    def set_world_params(self, v_max: float, a_max: float) -> None:
        """Sets the fixed parameters that characterize the world

        Sets the fixed parameters that define a given world, namely the speed limit
        and the maximum vehicle accererlation(deceleration)

        Parameters
        ----------
        v_max
            Maximum allowed/possible/desired velocity [m/s]
        a_max
            Maximum possible/desired acceleration and deceleration [m/s]
        """

        # Set v_max to the nearest smaller integer multiple of v_step
        self.v_max = round_to_step(v_max, self.v_step, behavior="floor")
        self.a_max = a_max

    def set_traffic_light_uniform_distribution(
        self, first_support: float, last_support: float
    ) -> None:
        """Sets the green light event distribution to the specified uniform distribution

        Sets the probability of event G occurring (the light turns green) for each timestep.
        The support boundaries indicate at what time event G has support (is nonzero).

        Parameters
        ----------
        first_support
            Time of first support of event G [s]
        last_support
            Time of last support of event G [s]
        """
        self.green_distribution = UniformDistribution(
            first_support, last_support, self.t_step
        )
        self.calc_t_eval(last_support)  # Calculate evaluation time

    def set_traffic_light_arbitrary_distribution(
        self,
        distribution: List[float],
    ) -> None:
        """Sets the green light event distribution to the specified distribution

        Sets the probability of event G occurring (the light turns green) for each timestep.

        Parameters
        ----------
        distribution
            List of floats indicating the probability of the light turning green at each timestep
        """
        self.green_distribution = ArbitraryDistribution(distribution, self.t_step)
        # TODO this is wrong MAYBE
        self.calc_t_eval(
            len(self.green_distribution.distribution) * self.t_step
        )  # Calculate evaluation time

    def calc_t_eval(self, last_support: float) -> None:
        """Calculates the time at which to measure reward

        Calculates the time at which the vehicle will definitely be back up to speed,
        after passing through the signalized intersection. This assumes that the traffic
        light distribution has a distinct time after which it has no support. This calculation
        takes acceleration and speed limit into account. At this time, we can compare the performance
        of any approach by looking at the vehicle position.

        Parameters
        ----------
        last_support
            Time at which event G will certainly have occurred [s]
        """
        # eval_gap is the time it takes to get up to top speed in worst case
        eval_gap = self.v_max / self.a_max
        self.t_eval = last_support + eval_gap

    def compute_state_space_shape(self) -> None:
        """Computes the size of the state space, and sets the state space edge boundaries"""
        num_v_steps = int((self.v_max - self.v_min) / self.v_step) + 1
        # Use abs() because x is negative
        num_x_steps = int(abs(self.x_max - self.x_min) / self.x_step) + 1
        self.state_space_shape = (num_x_steps, num_v_steps)
        self.state_space_bounds = (self.x_min, self.x_max, self.v_min, self.v_max)

    def set_init_conditions(self, x_start: float, v_start: float) -> None:
        """Sets the initial state of the vehicle, and computes state space meta info

        Parameters
        ----------
        x_start
            Initial position of vehicle [m]. This is always a negative number if the vehicle
            has not yet reached the intersection.
        v_start
            Initial velocity of vehicle [m/s]
        """
        assert x_start < 0, "x_start should be negative"
        self.x_min = round_to_step(x_start, self.x_step)
        v_start_discrete = round_to_step(v_start, self.v_step)
        self.compute_state_space_shape()  # Do this here so self.state_space_bounds are set
        self.initial_state = State(
            self.x_min, v_start_discrete, self.state_space_bounds
        )

    def configure(self, json_path: str) -> None:
        """Configures all parameters from a JSON file

        Configures compute, world, and traffic light parameters, sets initial conditions,
        and computes state space meta info.

        Parameters
        ----------
        json_path
            The file path where the configuration file exists
        """
        with open(json_path) as f:
            params = json.load(f)
        self.params = params  # This is for __repr__()
        compute_params = params["compute_params"]
        world_params = params["world_params"]
        traffic_light_params = params["traffic_light_params"]
        distribution_type = traffic_light_params["type"]
        init_conditions = params["init_conditions"]

        self.set_compute_params(
            x_step=compute_params["x_step"],
            v_step=compute_params["v_step"],
            t_step=compute_params["t_step"],
        )
        self.set_world_params(v_max=world_params["v_max"], a_max=world_params["a_max"])
        self.set_init_conditions(
            x_start=init_conditions["x_start"],
            v_start=init_conditions["v_start"],
        )

        # Set traffic light distribution
        if distribution_type == "uniform":
            self.set_traffic_light_uniform_distribution(
                first_support=traffic_light_params["first_support"],
                last_support=traffic_light_params["last_support"],
            )
        if distribution_type == "arbitrary":
            self.set_traffic_light_arbitrary_distribution(
                distribution=traffic_light_params["distribution"]
            )

    def discretize_state(self, state: State) -> State:
        """Discretizes a continuous valued state object to the nearest discrete step

        Parameters
        ----------
        state
            State object; continuous values allowed
        """
        new_x = round_to_step(state.x, self.x_step)
        new_v = round_to_step(state.v, self.v_step)

        # Use a new State object so that boundaries get checked
        return State(new_x, new_v, self.state_space_bounds)

    def state_to_indices(self, state: State) -> Tuple[int, int]:
        """Converts a State object to an index 2-tuple into the state space

        Parameters
        ----------
        state
            State object

        Returns
        -------
        tuple
            2-tuple of ints (x_idx, v_idx) that index into the state space
        """
        state = self.discretize_state(state)
        return (int(state.x / self.x_step * -1), int(state.v / self.v_step))

    def indices_to_state(self, indices: Tuple[int, int]) -> State:
        """Converts an index 2-tuple on the state space to a state

        Parameters
        ----------
        indices
            2-tuple of ints (x_idx, v_idx) that index into the state space

        Returns
        -------
        State
            A State object corresponding to the indices
        """
        return State(
            indices[0] * self.x_step * -1,
            indices[1] * self.v_step,
            self.state_space_bounds,
        )

    def timestep_to_time(self, timestep: int) -> float:
        """Converts an integer timestep to the corresponding time in seconds

        Parameters
        ----------
        timestep
            Index of discrete algorithm timestep

        Returns
        -------
        float
            Time [s]

        """
        return timestep * self.t_step

    def delta_x(self, state: State, v_new: float, mode: str = "trapezoidal") -> float:
        """Approximates the new vehicle position after one timestep has elapsed

        Approximates the new vehicle position based on its previous state and its new velocity.
        The integration method is given by `mode`.

        Parameters
        ----------
        state
            A State object indicating the state at timestep t
        v_new
            The vehicle velocity at timestep t+1 [m/s]
        mode
            A string indicating the integration method. One of ('trapeziodal','right')

        Returns
        -------
        float
            The approximate new position of the vehicle [m]

        Notes
        -----
        The integration mode 'right' uses a right Riemann sum, which is the default in SUMO.
        The integration mode 'trapezoidal' uses the trapezoidal rule, which is more accurate.
        SUMO can be set to use the trapezoidal rule to match this method, for better accuracy.
        """
        if mode == "trapezoidal":
            v_avg = (state.v + v_new) / 2
            x_new = state.x + v_avg * self.t_step
            x_new_discrete = round_to_step(x_new, self.x_step)
        if mode == "right":
            x_new = state.x + v_new * self.t_step
            x_new_discrete = round_to_step(x_new, self.x_step)
        return x_new_discrete

    @timer
    def build_adjacency_matrix(self) -> None:
        """Builds an adjacency matrix that indicates which states are reachable from a given state

        Builds an adjacency matrix of size state_space_shape x state_space_shape. Since the
        state space is 2D, the adjacency matrix is 4D. The "rows" (first two indices) represent
        states at t=k, and the "columns" (last two indices) represent states at t=k+1. The value
        of a cell is True if a vehicle can start from the "row" state at t=k and end at the
        "column" state at t=k+1, and False otherwise.

        Examples
        --------
        if self.adj_matrix[5,5,10,5] == True, this means that a vehicle at state indices [5,5] at
        t=k can reach state indices [10,5] at t=k+1
        """
        print("Building Adjacency Matrix")

        # Init boolean array
        ss_size = self.state_space_shape
        self.adj_matrix = np.zeros((ss_size * 2), dtype=np.bool_)

        # Iterate over all starting states
        for i, j in product(range(ss_size[0]), range(ss_size[1])):
            state = self.indices_to_state((i, j))
            a_increment = (
                self.a_max * self.t_step
            )  # max acceleration that can occur in a timestep
            v_min, v_max = (
                state.v - a_increment,
                state.v + a_increment,
            )  # min, max of reachable states
            v_min_discrete = round_to_step(v_min, self.v_step, behavior="ceil")
            v_max_discrete = round_to_step(v_max, self.v_step, behavior="floor")

            # Iterate over reachable velocities
            for v_new in np.arange(v_min_discrete, v_max_discrete, self.v_step):

                # Compute new position
                x_new_discrete = self.delta_x(state, v_new, mode="trapezoidal")

                # Set relevant element of adjacency matrix to True
                try:
                    new_state = State(x_new_discrete, v_new, self.state_space_bounds)
                except ValueError:  # If the state is out of bounds
                    pass
                else:
                    i_new, j_new = self.state_to_indices(new_state)
                    self.adj_matrix[
                        i, j, i_new, j_new
                    ] = True  # set appropriate edge to True

    def reward(self, state: State, timestep: int) -> float:
        """Returns the reward of a state

        Returns the reward of a given state and time, given that the traffic light is green
        at that time. It is the position of vehicle at t_eval, when it is definitely up to
        speed.

        Parameters
        ----------
        state
            Vehicle state
        timestep
            Index of discrete algorithm timestep

        Returns
        -------
        float
            The position at time t_eval

        References
        ----------
        Roy, Jonathan. Red Light Approach Notes. 3 April 2020
        """

        # delta_t: the difference between t_eval and the current time
        delta_t = self.t_eval - self.timestep_to_time(timestep)
        current_x = state.x
        max_delta_x = self.v_max * delta_t
        acceleration_loss = -1 / 2 * 1 / self.a_max * (self.v_max - state.v) ** 2
        return current_x + max_delta_x + acceleration_loss

    def reward_with_red_check(self, state: State, timestep: int) -> float:
        """Returns the reward of a state, with a low value if it leads to running the red light

        Returns the reward incurred from a given state and timestep if the light turns green
        on that timestep. However, it accounts for the possibility that the light does not
        turn green by giving a large negative reward if the vehicle will necessarily run
        the red light from the given state.

        Parameters
        ----------
        state
            Vehicle state
        timestep
            Index of discrete algorithm timestep

        Returns
        -------
        float
            The position at time t_eval, or a low value if running the red light

        Notes
        -----
        This reward function does not relax its negative penalty for running a red light, even if
        there is certainty that the light will turn green by a given timestep, implied by the green
        light distribution. This may be changed in the future.
        """
        reward = self.reward(state, timestep)

        # Check if you will run the red light at next timestep
        v_next_min = max(
            (state.v - self.a_max * self.t_step), self.v_min
        )  # maxed with v_min to avoid negative velocity
        x_next_min = state.x + self.delta_x(state, v_next_min, mode="trapezoidal")
        if x_next_min > 0:
            reward = -999999999  # very bad value
        return reward

    def find_max_next_state(
        self, state_indices: Tuple[int, int], timestep: int
    ) -> Tuple[float, int]:
        """Returns a pointer to the highest valued reachable state at the next timestep

        Given a state and timestep, this function finds the highest value reachable state at
        the next timestep. It returns a 2-tuple containing the value of and a pointer to that state.

        Parameters
        ----------
        state_indices
            Indices in the form (i, j) representing the state
        timestep
            Index of discrete algorithm timestep from which to look forward

        Returns
        -------
        tuple
            A 2-tuple with the value of and pointer to the highest value reachable state. Note that
            the pointer is ravelled; use np.unravel_index() to rectify
        """
        reachable_mask = self.adj_matrix[
            state_indices[0], state_indices[1]
        ]  # Take relevant slice of adj_matrix
        next_state_vals = np.zeros(self.state_space_shape)
        for i, j in product(
            range(self.state_space_shape[0]), range(self.state_space_shape[1])
        ):
            next_state_vals[i, j] = self.I[timestep + 1, i, j, 0]
        reachable_vals = reachable_mask * next_state_vals
        max_reachable = np.max(reachable_vals)
        argmax_reachable = np.argmax(reachable_vals)  # This stores the index raveled
        return (max_reachable, argmax_reachable)

    def calc_I(self, state_indices: Tuple[int, int], timestep: int) -> None:
        """Calculates the value of a given state at a given time

        Calculates the value of a given state at a given time based on the highest
        value state reachable at the next timestep, the probability of the light turning
        green at this timestep, and the reward of the state.

        Parameters
        ----------
        state_indices
            Indices in the form (i, j) representing the state
        timestep
            Index of discrete algorithm timestep at which to measure state value
        """

        # alpha is the probability of event G occurring right now
        alpha = self.green_distribution.distribution[timestep]
        if timestep != self.num_timesteps - 1:
            max_reachable, argmax_reachable = self.find_max_next_state(
                state_indices, timestep
            )
        else:
            max_reachable = 0
            argmax_reachable = 0  # This should never get used in the forward pass

        # Reward if light turns green now weighted by P(light turns green)
        weighted_reward = alpha * self.reward_with_red_check(
            self.indices_to_state(state_indices), timestep
        )

        # weighted expected value if light does not turn green right now
        weighted_long_term_value = (1 - alpha) * max_reachable

        # add up for total expected value
        value = weighted_reward + weighted_long_term_value

        # return expected value and pointer to next state
        return (value, argmax_reachable)

    @timer
    def backward_pass(self) -> None:
        """Calculates the value of every state at every timestep

        Iterates backwards from the last timestep to the first timestep, calculating
        the value of each state at each timestep. It must proceed backwards because
        the value of a state at t=k depends on the value of reachable states at t=k+1.
        """

        ss_shape = self.state_space_shape

        # calculate number of timesteps
        self.num_timesteps = len(
            self.green_distribution.distribution
        )  # This is the number of timesteps from 0 to last_support

        # Initialize I to timesteps x statespace size
        # Last dimension stores I and pointer to next state: [I, pointer_index]
        self.I = np.zeros((self.num_timesteps, ss_shape[0], ss_shape[1], 2))  # noqa

        # Iterate backwards through all timesteps to fill out I
        bar = IncrementalBar("Calculating I", max=self.num_timesteps)
        for timestep in range(self.num_timesteps - 1, -1, -1):

            # Iterate through all states
            for i, j in product(range(ss_shape[0]), range(ss_shape[1])):
                self.I[timestep, i, j] = self.calc_I((i, j), timestep)
            bar.next()
        bar.finish()

    def forward_step(self, state: State, timestep: int) -> Tuple[State, int]:
        """Returns the best reachable state at the next timestep

        Given a state and timestep t=k, return the State and ravelled index of the
        highest value reachable state at t=k+1.

        Parameters
        ----------
        state
            Vehicle state
        timestep
            Index of discrete algorithm timestep from which to look forward

        Returns
        -------
        tuple[State, int]
            Tuple containing the best reachable State, and its ravelled index
        """

        # Initial state indices
        i, j = self.state_to_indices(state)

        # Iterate forward through timesteps
        next_indices_raveled = int(self.I[timestep, i, j, 1])
        next_state_indices = np.unravel_index(
            next_indices_raveled, self.state_space_shape
        )
        next_state = self.indices_to_state(next_state_indices)

        return (next_state, next_state_indices)

    def forward_pass(self) -> List[State]:
        """Returns a motion plan defined by states at every timestep

        Starting at the initial state, calculates the optimal motion plan by following
        pointers in I() to find the best state at every timestep. The first element in
        the resulting list is the initial state, the second is the best reachable state
        at timestep 1, and so on.

        Returns
        -------
        List[State]
            A list of States defining the optimal motion plan
        """

        # Initialize list to store path through state space
        state_list = [self.initial_state]

        # Iterate forward through timesteps
        current_state = self.initial_state
        for timestep in range(1, self.num_timesteps):
            next_state, _ = self.forward_step(current_state, timestep)
            current_state = next_state
            state_list.append(current_state)

        return state_list
