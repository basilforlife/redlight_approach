import os
import sys

import argparse
import traci
import traci.constants as tc

from Red_Light_Approach.sumo_utils import add_sumo_path


# Set up argparse
parser = argparse.ArgumentParser(description='Run a SUMO simulation')
parser.add_argument('path_to_sumocfg', type=str, help='relative or absolute path to .sumocfg file to run')
parser.add_argument('-g', '--gui', action='store_true', help='Run the simulation in sumo-gui. Default is to run in sumo') 
args = parser.parse_args()

# Ensure sumo is on the system path
add_sumo_path()

# Set up sumo interfacing
if args.gui:
    sumoBinary = '/usr/local/bin/sumo-gui' # Use graphical simulator
else:
    sumoBinary = '/usr/local/bin/sumo' # Use command line only simulator
sumoCmd = [sumoBinary, '-c', args.path_to_sumocfg]

# Enter traci context
traci.start(sumoCmd)
traci.vehicle.subscribe('vehicle_0', (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION, tc.VAR_NEXT_TLS, tc.VAR_SPEED))
step = 0
print(f'traci simulation step size: {traci.simulation.getDeltaT()}')
while step < 50:
    traci.simulationStep()
    print('step', step)
    print(traci.vehicle.getSubscriptionResults('vehicle_0'))
    traci.vehicle.setSpeed('vehicle_0', 5)
    step += 1

# Exit traci context
traci.close()


