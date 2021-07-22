import pytest

from Red_Light_Approach.state import State


class TestApproach:
    def test_green_dist(self, approach, uniform_dist_result):
        assert (approach.green_dist.dist == uniform_dist_result).all()

    def test_calc_t_eval(self, approach, t_eval):
        assert approach.t_eval == t_eval 

    def test_init_state_v(self, approach, init_state):
        assert init_state.v == approach.initial_state.v

    def test_init_state_x(self, approach, init_state):
        assert init_state.x == approach.initial_state.x

    def test_compute_state_space_shape(self, approach, state_space_shape):
        assert approach.state_space_shape == state_space_shape

    def test_state_to_indices_to_state(self, approach, init_state):
        indices = approach.state_to_indices(init_state)
        new_state = approach.indices_to_state(indices)
        assert init_state == new_state


    def test_indices_to_state_to_indices(self, approach):
        indices = (100,36)
        new_state = approach.indices_to_state(indices)
        indices_2 = approach.state_to_indices(new_state)
        assert indices == indices_2
 
    def test_rho_red_light_check(self, approach, ran_red_state):
        assert approach.rho(ran_red_state,10) == -999999999

