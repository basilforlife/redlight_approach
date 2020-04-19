# Author: Jonathan Roy
# Date: 29 March 2020

# This script implements the red light approach optimization


# IMPORTS
import numpy as np


# UTILS

# This fn gives the value of a state at a point in time
# Args:
# state: a _____ containing the position and velocity
# delta_t: the difference between t_eval and the current time
# Returns: the value of a state if event G occurs at the passed time
def rho(state, delta_t, Vmax, Amax):
    pos, V = state
    result = pos + Vmax * delta_t - 1/2 * 1/Amax * (Vmax-V)**2

    # Check if you ran the red light
    if pos > 0: result = -999999999 # very bad value
    
    return result

# This fn tells if one state is reachable from a previous state
def isStateReachable(init_state, dest_state, Vmax, Amax, Tstep):
    # Fill in this logic
    return some_bool

# PARAMS
Vmax = 18 # max velocity car is permitted (m/s)
Amax = 4 # max acceleration car is permitted (m/s^2)

# Initial conditions
X = -150 # distance in meters from stoplight
t = 0 

# Set up time stuff
t_last = # the last timestep for which the dist. of event G has support
t_eval = t_last + Vmax/Amax # The time at which state values are eval'd

# Set up state space
Vstep = 0.05
Xstep = 0.5
Vmin = 0 # this is 0 for obvious reasons
Xmin = X # this is the initial position
Xmax = 0 # this is the value of X at the stoplight

SS = np.zeros(((Xmax-Xmin)/Xstep, (Vmax-Vmin)/Vstep))




# MAIN



# RUN MAIN
main()
