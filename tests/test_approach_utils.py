from math import isclose

from redlight_approach.approach_utils import round_to_step


class TestApproachUtils:
    def test_round_to_step_default(self):
        assert isclose(round_to_step(5.16, 0.1), 5.2)

    def test_round_to_step_ceil(self):
        assert isclose(round_to_step(5.14, 0.1, "ceil"), 5.2)

    def test_round_to_step_floor(self):
        assert isclose(round_to_step(5.16, 0.1, "floor"), 5.1)
