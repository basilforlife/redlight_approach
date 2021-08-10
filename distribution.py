from abc import ABC, abstractmethod
from math import floor, isclose
from typing import Any, List

import numpy as np


# Abstract base class for distribution
class Distribution(ABC):
    """Abstract Base Class for green light waiting time distribution"""

    def __init__(self, t_step: float) -> None:
        """Initialize t_step for discrete distribution

        Parameters
        ----------
        t_step
            The time step length in seconds
        """
        self.t_step = t_step

    @abstractmethod
    def __repr__(self, class_name: str, parameters: dict) -> str:
        """Returns a string with summary parameters of a distribution

        the __repr__ method is intended to be called from the method of the same name in a subclass,
        to standardize formatting of distribution summary parameters

        Parameters
        ----------
        class_name
            The name of the subclass
        parameters
            A dictionary containing names and values of summary parameters

        Returns
        -------
        str
            A string summarizing the object
        """
        return f"{class_name} object: parameters: {parameters}"

    @abstractmethod
    def sample(self) -> Any:
        """Sample from a continuous representation of the distribution

        Returns
        -------
        numpy.ndarray
            An array of samples
        """
        pass


class UniformDistribution(Distribution):
    """Represents a uniform distribution with a delay for green light wait times

    This class builds and stores a uniform distribution represented discretely
    with `t_step` step length, and provides a method `sample()` to sample
    continuously from the distribution
    """

    def __init__(
        self, first_support: float, last_support: float, t_step: float
    ) -> None:
        """Initialize UniformDistribution object and create self.dist member

        Parameters
        ----------
        first_support
            The earliest support of the green light distribution in seconds
        last_support
            The latest support of the green light distribution in seconds
        t_step
            The time step length in seconds

        Examples
        --------
        >>> UniformDistribution(2,4,1).distribution
        array([0. , 0. , 0.5, 0.5])
        """
        super().__init__(t_step)
        self.first_support = first_support
        self.last_support = last_support
        num_full_timesteps = floor(
            (self.last_support - self.first_support) / self.t_step
        )
        num_timesteps = floor(self.last_support / self.t_step)
        self.distribution = np.zeros(num_timesteps)
        support = np.ones(num_full_timesteps) / num_full_timesteps
        self.distribution[-num_full_timesteps:] = support

    def __repr__(self) -> str:
        """Returns a string with summary parameters of a uniform distribution

        Returns
        -------
        str
            A strings summarizing the object
        """
        params = {
            "first_support": self.first_support,
            "last_support": self.last_support,
            "t_step": self.t_step,
        }
        return super().__repr__("UniformDistribution", params)

    def sample(self, num_samples: int) -> Any:
        """Sample from a continuous representation of the distribution

        Parameters
        ----------
        num_samples
            The number of samples to take

        Returns
        -------
        numpy.ndarray
            A 1D array of samples from the distribution
        """
        self.rng = np.random.default_rng()
        return self.rng.uniform(self.first_support, self.last_support, num_samples)


class ArbitraryDistribution(Distribution):
    """Represents an arbitrary distribution with a delay for green light wait times

    This class builds and stores an arbitrary distribution represented discretely
    with `t_step` step length, and provides a method `sample()` to sample
    from the distribution
    """

    def __init__(self, distribution: List[float], t_step: float) -> None:
        """Initialize ArbitraryDistribution object and create self.dist member

        Parameters
        ----------
        distribution
        t_step
            The time step length in seconds
        """
        super().__init__(t_step)
        assert isclose(sum(distribution), 1), "Probability distribution must sum to 1"
        self.distribution = distribution

    def __repr__(self) -> str:
        """Returns a string describing the distribution

        Returns
        -------
        str
            A strings summarizing the object
        """
        params = {
            "distribution": self.distribution,
            "t_step": self.t_step,
        }
        return super().__repr__("ArbitraryDistribution", params)

    def sample(self, num_samples: int) -> Any:
        """Sample from a continuous representation of the distribution

        Parameters
        ----------
        num_samples
            The number of samples to take

        Returns
        -------
        numpy.ndarray
            A 1D array of samples from the distribution
        """
        rng = np.random.default_rng()
        index_list = rng.choice(
            list(range(len(self.distribution))), num_samples, self.distribution
        )
        return index_list * self.t_step
