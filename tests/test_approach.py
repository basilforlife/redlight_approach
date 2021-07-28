from Red_Light_Approach.approach import Approach


class TestApproach:
    def test_type(self, approach):
        assert isinstance(approach, Approach)

    def test_repr(self, approach):
        assert isinstance(approach.__repr__(), str)

    def test_eq(self, approach):
        assert approach == approach

    def test_calc_t_eval(self, approach, t_eval):
        assert approach.t_eval == t_eval

    def test_init_state(self, approach, init_state):
        assert init_state == approach.initial_state

    def test_compute_state_space_shape(self, approach, state_space_shape):
        assert approach.state_space_shape == state_space_shape

    def test_state_to_indices_to_state(self, approach, init_state):
        indices = approach.state_to_indices(init_state)
        new_state = approach.indices_to_state(indices)
        assert init_state == new_state

    def test_indices_to_state_to_indices(self, approach):
        indices = (100, 36)
        new_state = approach.indices_to_state(indices)
        indices_2 = approach.state_to_indices(new_state)
        assert indices == indices_2

    def test_rho_red_light_check(self, approach, ran_red_state):
        assert approach.rho(ran_red_state, 10) == -999999999
