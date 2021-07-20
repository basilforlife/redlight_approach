import argparse
import os
import pickle
from random import uniform
import sys
import xml.etree.ElementTree as ET

import traci
import traci.constants as tc

from Red_Light_Approach.approach import Approach
from Red_Light_Approach.state import State 
from Red_Light_Approach.sumo_utils import add_sumo_path


# Do xml parsing to get timeloss results
# Positive number represents approach having a beneficial effect
def get_timeloss_diff(filepath):
    timeloss_dict = get_timeloss_dict(filepath)
    timeloss_vehicle_0 = timeloss_dict['vehicle_0']
    timeloss_vehicle_1 = timeloss_dict['vehicle_1']
    timeloss_diff = timeloss_vehicle_1 - timeloss_vehicle_0
    return timeloss_diff

def get_timeloss_dict(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    timeloss_dict = {}
    for child in root:
        timeloss_dict[child.attrib['id']] = float(child.attrib['timeLoss'])
    return timeloss_dict
   


# this fn gets random time and sets the light to red for that long
def set_red_light_random():
    red_duration = uniform(0, 10) + 15.228# random value plus a constant 
    traci.trafficlight.setPhase('0', 2) # set traffic light id=2 to red
    traci.trafficlight.setPhaseDuration('0', red_duration)
    print(f'Red light duration = {red_duration}')
    return red_duration

# This fn puts a vehicle in the left lane permanently
def set_lane_change_static(vehicle_ID):
    traci.vehicle.setLaneChangeMode(vehicle_ID, 0x000000000000)

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
step = 0
begin_approach = False
end_approach = False
approach_timestep = 0
while step < 50:
    print('step', step)
    sub_results = traci.vehicle.getSubscriptionResults('vehicle_0')
    sub_results_1 = traci.vehicle.getSubscriptionResults('vehicle_1')

    if not sub_results and not sub_results_1:
        break
    # check that traffic light is ahead (that next_TLS result is full)
    next_TLS = sub_results[112]
    green_light = red_duration < step + 1# This is true if light is green
    if next_TLS: # if traffic light ahead
        # TODO make state check boundaries to ensure state is within state space
        state = State(next_TLS[0][2] * -1, sub_results[64])
        print(f'green_light = {green_light}')

        # Begin approach at 100m when we can "see" it
        if state.x >= -100 and not begin_approach:
            begin_approach = True
            traci.vehicle.setColor('vehicle_0', (246,186,34)) # Change color when approach starts

    print(sub_results)
    print(sub_results_1)

    # End approach if it has started and the light is now green
    if begin_approach and green_light:
        end_approach = True
        traci.vehicle.setSpeed('vehicle_0', -1) # Hand control back to sumo
        traci.vehicle.setColor('vehicle_0', (24,255,0)) # Change color when approach ends

    if begin_approach and not end_approach:
        next_state, _ = approach.forward_step(state, approach_timestep)
        traci.vehicle.setSpeed('vehicle_0', next_state.v)
        print(f'next_state = {next_state}')
        approach_timestep += 1
       
    traci.simulationStep()
    step += 1

# Exit traci context
traci.close()


# Report timeloss
print(f'Red light duration = {red_duration}')
print(f'Time Diff = {get_timeloss_diff("sumo/two_roads/tripinfo.xml")}s')
