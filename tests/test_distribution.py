import pytest


class TestDistribution:
    def test_uniform_dist(self, uniform_dist, uniform_dist_result):
        assert (uniform_dist.dist == uniform_dist_result).all()
