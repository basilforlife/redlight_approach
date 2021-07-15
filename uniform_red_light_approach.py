import argparse
import os
import sys

import traci
import traci.constants as tc

from Red_Light_Approach.approach import Approach
from Red_Light_Approach.state import State 
from Red_Light_Approach.sumo_utils import add_sumo_path


# Set up argparse
parser = argparse.ArgumentParser(description='Run a SUMO simulation with an approach controller')
parser.add_argument('path_to_sumocfg', type=str, help='relative or absolute path to .sumocfg file to run')
parser.add_argument('path_to_approach_config', type=str, help='relative or absolute path to .json file to configure approach')
parser.add_argument('-g', '--gui', action='store_true', help='Run the simulation in sumo-gui. Default is to run in sumo') 
args = parser.parse_args()

# Ensure sumo is on system path
add_sumo_path()

# Set up sumo interfacing
if args.gui:
    sumoBinary = '/usr/local/bin/sumo-gui' # Use graphical simulator
else:
    sumoBinary = '/usr/local/bin/sumo' # Use command line only simulator
sumoCmd = [sumoBinary, '-c', args.path_to_sumocfg]


# Set up approach stuff
approach = Approach(args.path_to_approach_config)
approach.build_adjacency_matrix()
approach.backward_pass()


# Enter traci context
traci.start(sumoCmd)
traci.vehicle.subscribe('vehicle_0', (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_NEXT_TLS))
step = 0
print(f'traci simulation step size: {traci.simulation.getDeltaT()}')
begin_approach = False
while step < 50:
    traci.simulationStep()
    print('step', step)
    sub_results = traci.vehicle.getSubscriptionResults('vehicle_0')
    # TODO make state check boundaries to ensure state is within state space
    state = State(sub_results[112][0][2] * -1, sub_results[64])
    print(f'state = {state}')

    # Begin approach at 100m when we can "see" it
    if state.x >= -100 and not begin_approach:
        begin_approach = True
        approach_timestep = 0
        traci.vehicle.setColor('vehicle_0', (246,186,34)) # Change color when approach starts

    if begin_approach:
        next_state, _ = approach.forward_step(state, approach_timestep)
        traci.vehicle.setSpeed('vehicle_0', next_state.v)
        print('Approach modified speed')
        print(f'next_state = {next_state}')
       
    step += 1

# Exit traci context
traci.close()


