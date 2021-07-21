import argparse
from random import uniform

import traci
import traci.constants as tc

from Red_Light_Approach.state import State 
from Red_Light_Approach.sumo_utils import *


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
traci.start(sumo_cmd)
traci.vehicle.subscribe('vehicle_0', (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_NEXT_TLS))
traci.vehicle.subscribe('vehicle_1', (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_NEXT_TLS))
delay = 5.228 # Time to get to 100m
red_duration = set_red_light_uniform_random(delay+10, delay+20)

# Looping things
step = 0
approaching = False
while traci.simulation.getMinExpectedNumber() != 0:
    traci.simulationStep()
    step += 1
    sub_results = traci.vehicle.getSubscriptionResults('vehicle_0')
    sub_results_1 = traci.vehicle.getSubscriptionResults('vehicle_1')
    green_light = red_duration < step + 1# This is true if light is green

    print('step', step)
    print(f'green_light = {green_light}')
    print(sub_results)
    print(sub_results_1)
 
    # Check to see if vehicle_0 has a traffic light ahead, else continue
    # Everything below here in the while loop is approach control
    try:
        next_TLS = sub_results[112][0]
    except (KeyError, IndexError):
        continue 

    # Statements in this block run if there is a traffic light ahead
    # -----------------BEGIN TLS-----------------------------------

    # Extract state from subscription results
    state = State(next_TLS[2] * -1, sub_results[64])

    # Begin approach
    # This runs only once when vehicle arrives in state space bounds
    if state.x >= -100 and not green_light and not approaching:
        approach_timestep = 0
        approaching = True
        traci.vehicle.setColor('vehicle_0', (246,186,34)) # Change color when approach starts

    # End approach
    # This runs only once to end the approach control
    if green_light and approaching:
        approaching = False
        traci.vehicle.setSpeed('vehicle_0', -1) # Hand control back to sumo
        traci.vehicle.setColor('vehicle_0', (24,255,130)) # Change color when approach ends

    # This runs every timestep to control the approach
    if approaching:
        next_state, _ = approach.forward_step(state, approach_timestep)
        traci.vehicle.setSpeed('vehicle_0', next_state.v)
        approach_timestep += 1

    # -------------------END TLS-----------------------------------

# Exit traci context
traci.close()

# Report timeloss
print(f'Red light duration = {red_duration}')
print(f'Time Diff = {get_timeloss_diff("sumo/two_roads/tripinfo.xml", "vehicle_0", "vehicle_1")}s')
