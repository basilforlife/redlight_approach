from math import isclose

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
        assert isclose(sum(uniform_dist.distribution), 1)

    def test_sample(self, uniform_dist):
        N = 10
        assert len(uniform_dist.sample(N)) == N

    def test_leading_zeros(self, uniform_dist):
        result = uniform_dist.distribution[0:9]
        assert (result == np.zeros_like(result)).all()

    def test_support(self, uniform_dist):
        assert len([s for s in uniform_dist.distribution if s > 0]) == 10

    def test_interpolate_len(self, uniform_dist_interpolated):
        assert len(uniform_dist_interpolated.distribution) == 20

    def test_interpolate_val(self, uniform_dist_interpolated):
        assert isclose(uniform_dist_interpolated.distribution[-1], 0.1)


class TestArbitraryDistribution:
    def test_abc_type(self, arbitrary_dist):
        assert isinstance(arbitrary_dist, Distribution)

    def test_type(self, arbitrary_dist):
        assert isinstance(arbitrary_dist, ArbitraryDistribution)

    def test_repr(self, arbitrary_dist):
        assert isinstance(arbitrary_dist.__repr__(), str)

    def test_dist_sum(self, arbitrary_dist):
        assert isclose(sum(arbitrary_dist.distribution), 1)

    def test_wrong_sum_fails(self):
        with pytest.raises(AssertionError):
            ArbitraryDistribution([0, 0, 0.5, 0.4], "match", 1)

    def test_sample(self, arbitrary_dist):
        N = 10
        assert len(arbitrary_dist.sample(N)) == N

    def test_interpolate_len(self, arbitrary_dist_interpolated):
        assert len(arbitrary_dist_interpolated.distribution) == 4

    def test_interpolate_val(self, arbitrary_dist_interpolated):
        v = [0, 0.2, 0.6, 0.2]
        assert all(
            isclose(arbitrary_dist_interpolated.distribution[i], v[i]) for i in range(4)
        )
