import math

import numpy as np
import pytest

from redlight_approach.distribution import (
    ArbitraryDistribution,
    Distribution,
    UniformDistribution,
)


class TestUniformDistribution:
    def test_abc_type(self, uniform_dist):
        assert isinstance(uniform_dist, Distribution)

    def test_type(self, uniform_dist):
        assert isinstance(uniform_dist, UniformDistribution)

    def test_repr(self, uniform_dist):
        assert isinstance(uniform_dist.__repr__(), str)

    def test_dist_sum(self, uniform_dist):
        assert math.isclose(sum(uniform_dist.distribution), 1)

    def test_sample(self, uniform_dist):
        N = 10
        assert len(uniform_dist.sample(N)) == N

    def test_leading_zeros(self, uniform_dist):
        result = uniform_dist.distribution[0:9]
        assert (result == np.zeros_like(result)).all()

    def test_support(self, uniform_dist):
        result = uniform_dist.distribution[10:19]
        assert (result == np.zeros_like(result) + 0.1).all()


class TestArbitraryDistribution:
    def test_abc_type(self, arbitrary_dist):
        assert isinstance(arbitrary_dist, Distribution)

    def test_type(self, arbitrary_dist):
        assert isinstance(arbitrary_dist, ArbitraryDistribution)

    def test_repr(self, arbitrary_dist):
        assert isinstance(arbitrary_dist.__repr__(), str)

    def test_dist_sum(self, arbitrary_dist):
        assert math.isclose(sum(arbitrary_dist.distribution), 1)

    def test_wrong_sum_fails(self):
        with pytest.raises(AssertionError):
            ArbitraryDistribution([0, 0, 0.5, 0.4], 1)

    def test_sample(self, arbitrary_dist):
        N = 10
        assert len(arbitrary_dist.sample(N)) == N
