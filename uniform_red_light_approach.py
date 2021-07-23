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
parser.add_argument('-s', '--sumocfg_file',
                    default='sumo/two_roads/f.sumocfg',
                    type=str,
                    help='path to .sumocfg file to run. Default sumo/two_roads/f.sumocfg')
parser.add_argument('-c', '--approach_config_file',
                    type=str,
                    help='path to .json file to configure approach')
parser.add_argument('-p', '--pickle',
                    type=str,
                    help='File to pickle approach object to')
parser.add_argument('-u', '--unpickle',
                    type=str,
                    help='File to load approach object pickle from')
parser.add_argument('-g', '--gui',
                    action='store_true',
                    help='Run the simulation in sumo-gui. Default is to run in sumo') 
args = parser.parse_args()

# Ensure sumo is on system path
add_sumo_path()

# Load approach object according to command line args
approach = load_approach(args)

# Configure sumo to match approach params
speed_limit = approach.v_max 
approach_distance = approach.x_min
route_filename = 'sumo/two_roads/f.rou.xml'
new_route_filename = 'sumo/two_roads/modified.rou.xml'
set_accel_decel_xml_file(approach.a_max, approach.a_max, route_filename, new_route_filename)

# Configure sumo command with all the flags we need
sumo_cmd = sumo_command(args, approach, new_route_filename)

# Set up simulation stuff
delay = 5.228 # Time to get to 100m
num_samples = 30 
red_durations = np.random.uniform(delay+10, delay+20, num_samples) # random sample from uniform distribution
timeloss = np.zeros_like(red_durations)

# Run simulation
for i, red_duration in enumerate(red_durations):
    timeloss[i] = run_sumo(sumo_cmd, approach, red_duration, speed_limit, approach_distance)
    
# Cleanup
os.remove(new_route_filename)

# Plot
#plot_results(red_durations, timeloss)
plt.scatter(red_durations, timeloss)
#plt.axvline(0)
#plt.axhline(0)
plt.title('test')
plt.show()

# Print avg time saved
print(f'Mean time saved: {np.mean(timeloss)}s')

