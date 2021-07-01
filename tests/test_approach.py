import pytest


class TestDistribution:	
    def test_uniform_dist(self, uniform_dist, uniform_dist_result):
        assert (uniform_dist.dist == uniform_dist_result).all()


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
        assert new_state == init_state

    def test_build_adjacency_matrix(self, approach):
        approach.build_adjacency_matrix()
        assert True

