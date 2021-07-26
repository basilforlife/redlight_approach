from abc import ABC, abstractmethod
from math import floor, ceil
import numpy as np


# Abstract base class for distribution
class Distribution(ABC):

    def __init__(self, t_step):
        self.t_step = t_step
        self.rng = np.random.default_rng()

    @abstractmethod
    def __repr__(self, class_name, parameters):
        return f'{class_name} object: parameters: {parameters}'

    @abstractmethod
    def sample(self):
        pass

# Uniform distribution with delay
class UniformDistribution(Distribution):

    def __init__(self, first_support, last_support, t_step):
        super().__init__(t_step)
        self.first_support = first_support
        self.last_support = last_support
        num_full_timesteps = floor((self.last_support - self.first_support)/self.t_step)
        num_timesteps = floor(self.last_support/self.t_step)
        self.dist = np.zeros(num_timesteps)
        support = np.ones(num_full_timesteps)/num_full_timesteps
        self.dist[-num_full_timesteps:] = support

    def __repr__(self):
        params = {'first_support': self.first_support,
                  'last_support' : self.last_support,
                  't_step'       : self.t_step}
        return super().__repr__('UniformDistribution', params)

    # This returns continuous samples with units [s]
    def sample(self, num_samples):
        return self.rng.uniform(self.first_support, self.last_support, num_samples)
