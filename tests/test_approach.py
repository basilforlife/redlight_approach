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

    def test_idx_state_conversions(self, approach, init_state):
        idx = approach.state_to_idx(init_state)
        state = approach.idx_to_state(idx)
        assert state.x == init_state.x and state.v == init_state.v

    def test_state_space_flat_size(self, approach):
        ss_flat_size = approach.state_space_shape[0] * approach.state_space_shape[1]
        assert approach.state_space_flat_size == ss_flat_size

    def test_build_adjacency_matrix(self, approach):
        approach.build_adjacency_matrix()
        assert True

