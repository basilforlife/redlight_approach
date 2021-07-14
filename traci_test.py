import os
import sys

import argparse
import traci
import traci.constants as tc


# Set up argparse
parser = argparse.ArgumentParser(description='Run a SUMO simulation')
parser.add_argument('-g', '--gui', action='store_true', help='Run the simulation in sumo-gui. Default is to run in sumo') 
args = parser.parse_args()

# Ensure sumo is on the python load path
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# Set up sumo interfacing
if args.gui:
    sumoBinary = '/usr/local/bin/sumo-gui' # Use graphical simulator
else:
    sumoBinary = '/usr/local/bin/sumo' # Use command line only simulator
sumoCmd = [sumoBinary, '-c', 'sumo/simple_traffic_light/simple_traffic_light.sumocfg']

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


