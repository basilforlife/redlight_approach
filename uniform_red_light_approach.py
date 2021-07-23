import argparse
import numpy as np
import os

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

# Configure sumo to match approach params
speed_limit = approach.v_max 
route_filename = 'sumo/two_roads/f.rou.xml'
new_route_filename = 'sumo/two_roads/modified.rou.xml'
set_accel_decel_xml_file(approach.a_max, approach.a_max, route_filename, new_route_filename)

# Configure sumo command with all the flags we need
sumo_cmd = sumo_command(args, approach, new_route_filename)

# Set up simulation stuff
delay = 5.228 # Time to get to 100m
num_samples = 1 
red_durations = np.random.uniform(delay+10, delay+20, num_samples) # random sample from uniform distribution
timeloss = np.zeros_like(red_durations)

# Run simulation
for i, red_duration in enumerate(red_durations):
    timeloss[i] = run_sumo(sumo_cmd, approach, red_duration, speed_limit)
    
# Cleanup
os.remove(new_route_filename)

plt.scatter(red_durations, timeloss)
plt.show()
