import argparse
import pickle
from random import uniform

import traci
import traci.constants as tc

from Red_Light_Approach.approach import Approach
from Red_Light_Approach.state import State 
from Red_Light_Approach.sumo_utils import *


# this fn gets random time and sets the light to red for that long
def set_red_light_random():
    red_duration = uniform(0, 10) + 15.228# random value plus a constant 
    traci.trafficlight.setPhase('0', 2) # set traffic light id=2 to red
    traci.trafficlight.setPhaseDuration('0', red_duration)
    return red_duration

# Set up argparse
parser = argparse.ArgumentParser(description='Run a SUMO simulation with an approach controller')
parser.add_argument('path_to_sumocfg', type=str, help='relative or absolute path to .sumocfg file to run')
parser.add_argument('path_to_approach_config', type=str, help='relative or absolute path to .json file to configure approach')
parser.add_argument('-g', '--gui', action='store_true', help='Run the simulation in sumo-gui. Default is to run in sumo') 
parser.add_argument('-p', '--pickle', action='store_true', help='Pickle approach object')
parser.add_argument('-u', '--unpickle', action='store_true', help='Load approach object from pickle')
args = parser.parse_args()

# Ensure sumo is on system path
add_sumo_path()

# Set up sumo interfacing
if args.gui:
    sumoBinary = '/usr/local/bin/sumo-gui' # Use graphical simulator
else:
    sumoBinary = '/usr/local/bin/sumo' # Use command line only simulator
sumoCmd = [sumoBinary, '-c', args.path_to_sumocfg]


# pickle approach bc its slow to build
if args.pickle:
    # Set up approach stuff
    approach = Approach(args.path_to_approach_config)
    approach.build_adjacency_matrix()
    approach.backward_pass()

    # pickle results
    with open('approach.pickle', 'wb') as f:
        pickle.dump(approach, f)

# Unpickle if called for
if args.unpickle:
    with open('approach.pickle', 'rb') as f:
        approach = pickle.load(f)

# Enter traci context
traci.start(sumoCmd)
traci.vehicle.subscribe('vehicle_0', (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_NEXT_TLS))
traci.vehicle.subscribe('vehicle_1', (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_NEXT_TLS))
red_duration = set_red_light_random()

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
