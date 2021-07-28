import math

import pytest


class TestUniformDistribution:
    def test_dist_sum(self, uniform_dist):
        assert math.isclose(sum(uniform_dist.dist), 1)

    def test_sample(self, uniform_dist):
        N = 10
        assert len(uniform_dist.sample(N)) == N

    def test_repr(self, uniform_dist):
        assert type(uniform_dist.__repr__()) == str
