import numbers
from abc import ABC, abstractmethod
from math import isclose
from typing import Any, List, Union

import numpy as np
from scipy.interpolate import interp1d as interpolate


# Abstract base class for distribution
class Distribution(ABC):
    """Abstract Base Class for green light waiting time distribution"""

    @abstractmethod
    def __init__(
        self, specification_t_step: Union[float, str], representation_t_step: float
    ) -> None:
        """Initialize discrete distribution object and interpolate

        Parameters
        ----------
        specification_t_step
            One of "match" or a float. if "match", this value is set equal to
            `representation_t_step`. If a float, it is the timetep length used to specify the
            parameters of the distribution or the distribution itself [s].
        representation_t_step
            The timestep length used to represent the distribution [s]
        """
        self.__set_t_step_members(specification_t_step, representation_t_step)
        self.__interpolate_distribution()

    def __repr__(self) -> str:
        """Returns a string with summary parameters of a distribution

        the __repr__ method is intended to be called from the method of the same name in a subclass,
        to standardize formatting of distribution summary parameters

        Returns
        -------
        str
            A string summarizing the object
        """
        params = {
            "distribution": list(self.distribution),
            "specification_t_step": self.specification_t_step,
            "representation_t_step": self.representation_t_step,
        }
        return f"{type(self).__name__} object: parameters: {params}"

    def __set_t_step_members(
        self, specification_t_step: Union[float, str], representation_t_step: float
    ) -> None:
        """Sets the t_step object members after processing special cases

        Parameters
        ----------
        specification_t_step
            One of "match" or a float. if "match", this value is set equal to
            `representation_t_step`. If a float, it is the timetep length used to specify the
            parameters of the distribution or the distribution itself [s].
        representation_t_step
            The timestep length used to represent the distribution [s]

        """
        if specification_t_step == "match":
            self.specification_t_step = representation_t_step
        else:
            assert isinstance(
                specification_t_step, numbers.Number
            ), "'specification_t_step' must be 'match' or a float"
            self.specification_t_step = specification_t_step
        self.representation_t_step = representation_t_step

    def __interpolate_distribution(self) -> None:
        """Interpolates distribution to `representation_t_step` indices

        Replaces the `distribution` member, which is specified with `specification_t_step` indices,
        with a distribution interpolated to be in terms of `representation_t_step`.
        """
        distribution = list(self.distribution)

        # Define the ratio at which to interpolate
        t_step_ratio = self.representation_t_step / self.specification_t_step

        # Define a PDF (not a true PDF; it is relative) over the distribution's support,
        # expressed with specification_t_step sized time units. Appending zero and extrapolation
        # are necessary because each point in the given PMF('distribution' member) represents
        # the cumulative probabilty of the event occuring during the timestep starting at the
        # time corresponding to the point. This means the distribution has support for one
        # timestep beyond the last point in the given PMF. Interpolation using 'previous' corresponds
        # to this way of representing a PDF with a PMF.
        distribution.append(0)
        relative_PDF = interpolate(
            range(len(distribution)),
            distribution,
            kind="previous",
            fill_value="extrapolate",
        )

        # Define a range in terms of specification_t_step to access
        # values at every representation_t_step
        steps = np.arange(0, len(distribution) - 1, t_step_ratio)

        # Define a function mapping x to the average value of relative_PDF from x to x + step
        def average(x):
            return np.mean(relative_PDF(np.arange(x, x + t_step_ratio, 1e-7)))

        # Interpolate
        new_distribution = [average(timestep) for timestep in steps]

        # Normalize so sum = 1
        new_distribution /= np.sum(new_distribution)

        self.distribution = new_distribution

    def sample(self, num_samples: int) -> Any:
        """Sample timesteps from the distribution

        Samples durations using the distribution over durations, as defined in
        each child class instance, to weight the choices.

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
        random_indices = rng.choice(
            len(self.distribution), size=num_samples, p=self.distribution
        )
        return (
            random_indices * self.representation_t_step
        )  # return results with units of seconds, not timesteps


class UniformDistribution(Distribution):
    """Represents a uniform distribution with a delay for green light wait times

    This class builds and stores a uniform distribution represented discretely
    with `representation_t_step` step length, and provides a method `sample()` to sample
    from the distribution
    """

    def __init__(
        self,
        first_support: int,
        last_support: int,
        specification_t_step: Union[str, float],
        representation_t_step: float,
    ) -> None:
        """Initialize UniformDistribution object with distribution parameters

        This creates a discrete representation of a uniform distribution on the half open
        interval [first_support, last_support). It is specified in terms of `specification_t_step`,
        and converted to a representation in terms of `representation_t_step`

        Parameters
        ----------
        first_support
            The earliest support of the green light distribution, in terms of `specification_t_step`
        last_support
            The latest support of the green light distribution
        specification_t_step
            One of "match" or a float. if "match", this value is set equal to
            `representation_t_step`. If a float, it is the timetep length used to specify the
            parameters of the distribution or the distribution itself [s].
        representation_t_step
            The timestep length used to represent the distribution [s]

        Examples
        --------
        >>> UniformDistribution(2,4,1).distribution
        array([0. , 0. , 0.5, 0.5])
        """
        num_full_timesteps = last_support - first_support
        num_timesteps = last_support
        self.distribution = np.zeros(num_timesteps)
        support = np.ones(num_full_timesteps) / num_full_timesteps
        self.distribution[-num_full_timesteps:] = support
        super().__init__(specification_t_step, representation_t_step)


class ArbitraryDistribution(Distribution):
    """Represents an arbitrary probability distribution discretely

    This class builds and stores an arbitrary distribution represented discretely
    with `representation_t_step` step length, and provides a method `sample()` to sample
    from the distribution
    """

    def __init__(
        self,
        distribution: List[float],
        specification_t_step: Union[str, float],
        representation_t_step: float,
    ) -> None:
        """Initialize ArbitraryDistribution object with discrete distribution

        Parameters
        ----------
        distribution
            A list specifying the probability distribution of an event occuring
        specification_t_step
            One of "match" or a float. if "match", this value is set equal to
            `representation_t_step`. If a float, it is the timetep length used to specify the
            parameters of the distribution or the distribution itself [s].
        representation_t_step
            The timestep length used to represent the distribution [s]
        """
        assert isclose(sum(distribution), 1), "Probability distribution must sum to 1"
        self.distribution = distribution
        super().__init__(specification_t_step, representation_t_step)
