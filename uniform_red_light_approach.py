import argparse
import numpy as np

import matplotlib.pyplot as plt
import traci
import traci.constants as tc

from Red_Light_Approach.state import State 
from Red_Light_Approach.sumo_utils import *
from Red_Light_Approach.run_sumo import run_sumo


# Set up argparse
parser = argparse.ArgumentParser(description='Run a SUMO simulation with an approach controller')
parser.add_argument('path_to_sumocfg', type=str, help='relative or absolute path to .sumocfg file to run')
parser.add_argument('path_to_approach_config', type=str, help='relative or absolute path to .json file to configure approach')
parser.add_argument('-g', '--gui', action='store_true', help='Run the simulation in sumo-gui. Default is to run in sumo') 
parser.add_argument('-p', '--pickle', action='store_true', help='Pickle approach object')
parser.add_argument('-u', '--unpickle', action='store_true', help='Load approach object from pickle')
parser.add_argument('--pickle-file', default='approach.pickle', type=str, help='Location of pickle file. Defaults to approach.pickle')
args = parser.parse_args()

# Ensure sumo is on system path
add_sumo_path()

# Load approach object according to command line args
approach = load_approach(args)

# Enter traci context
sumo_cmd = sumo_command(args)
delay = 5.228 # Time to get to 100m
num_samples = 5 
red_durations = np.random.uniform(delay+10, delay+20, num_samples) # random sample from uniform distribution
timeloss = np.zeros_like(red_durations)
# Run simulation
for i, red_duration in enumerate(red_durations):
    timeloss[i] = run_sumo(sumo_cmd, approach, red_duration)
    
plt.scatter(red_durations, timeloss)
plt.show()
