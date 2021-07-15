from math import floor, ceil
import numpy as np

# Dist class holds a distribution representing the green light event
class Distribution:
    def __init__(self):
        self.dist = None

    # Times are with reference to current time
    def uniform_dist(self, first_support, last_support, t_step):
        num_full_timesteps = floor((last_support - first_support)/t_step)
        num_timesteps = floor(last_support/t_step)
        self.dist = np.zeros(num_timesteps)
        support = np.ones(num_full_timesteps)/num_full_timesteps
        self.dist[-num_full_timesteps:] = support
