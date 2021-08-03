from math import isclose

import pytest

from redlight_approach.approach import Approach
from redlight_approach.state import State


class TestApproach:
    def test_type(self, approach):
        assert isinstance(approach, Approach)

    def test_repr(self, approach):
        assert isinstance(approach.__repr__(), str)

    def test_eq(self, approach):
        with pytest.raises(NotImplementedError):
            approach == approach

    def test_set_world_params(self):
        approach = Approach()
        approach.set_compute_params(0.5, 0.5, 1)
        approach.set_world_params(10.4, 2.5)
        assert isclose(approach.v_max, 10)

    def test_calc_t_eval(self, approach, t_eval):
        assert approach.t_eval == t_eval

    def test_init_state(self, approach, init_state):
        assert init_state == approach.initial_state

    def test_compute_state_space_shape(self, approach, state_space_shape):
        assert approach.state_space_shape == state_space_shape

    def test_discretize_state(self, approach, init_state):
        state = State(-99.55, 17.76)
        assert approach.discretize_state(state) == init_state

    def test_state_to_indices_to_state(self, approach, init_state):
        indices = approach.state_to_indices(init_state)
        new_state = approach.indices_to_state(indices)
        assert init_state == new_state

    def test_indices_to_state_to_indices(self, approach):
        indices = (100, 36)
        new_state = approach.indices_to_state(indices)
        indices_2 = approach.state_to_indices(new_state)
        assert indices == indices_2

    def test_reward(self, approach, init_state):
        reward = approach.reward(init_state, 0)
        assert isclose(reward, 389.6)  # -100 + 27.2 * 18

    def test_reward_2(self, approach):
        reward = approach.reward(State(-50, 10), 20)
        assert isclose(reward, 66.8)  # -50 + (27.2 - 20) * 18 - (1/2)*(1/2.5)*(18-10)^2

    def test_reward_with_red_check(self, approach, ran_red_state):
        assert approach.reward_with_red_check(ran_red_state, 10) == -999999999

    def test_reward_with_red_check_2(self, approach, will_run_red_state):
        assert approach.reward_with_red_check(will_run_red_state, 10) == -999999999

    def test_timestep_to_time(self, approach):
        approach.t_step = 0.5
        assert approach.timestep_to_time(17) == 8.5
