import math

import numpy as np

from Red_Light_Approach.distribution import Distribution, UniformDistribution


class TestUniformDistribution:
    def test_abc_type(self, uniform_dist):
        assert isinstance(uniform_dist, Distribution)

    def test_type(self, uniform_dist):
        assert isinstance(uniform_dist, UniformDistribution)

    def test_dist_sum(self, uniform_dist):
        assert math.isclose(sum(uniform_dist.dist), 1)

    def test_sample(self, uniform_dist):
        N = 10
        assert len(uniform_dist.sample(N)) == N

    def test_repr(self, uniform_dist):
        assert isinstance(uniform_dist.__repr__(), str)

    def test_leading_zeros(self, uniform_dist):
        result = uniform_dist.dist[0:9]
        assert (result == np.zeros_like(result)).all()

    def test_support(self, uniform_dist):
        result = uniform_dist.dist[10:19]
        assert (result == np.zeros_like(result) + 0.1).all()
