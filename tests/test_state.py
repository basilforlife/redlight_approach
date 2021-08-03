import pytest

from redlight_approach.state import State


class TestState:
    def test_type(self, init_state):
        assert isinstance(init_state, State)

    def test_repr(self, init_state):
        assert isinstance(init_state.__repr__(), str)

    def test_eq(self, init_state):
        same_state = State(init_state.x, init_state.v)
        assert init_state == same_state

    def test_x_boundaries(self, state_bounds):
        with pytest.raises(ValueError):
            State(1, 10, state_bounds)

    def test_v_boundaries(self, state_bounds):
        with pytest.raises(ValueError):
            State(1, 100, state_bounds)

    def test_valid_boundaries(self, state_bounds):
        assert State(-10, 10, state_bounds)
