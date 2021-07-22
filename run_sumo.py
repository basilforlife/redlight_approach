import traci
import traci.constants as tc

from Red_Light_Approach.state import State
from Red_Light_Approach.sumo_utils import *


# This fn runs a sumo/traci simulation and returns the timeLoss difference
def run_sumo(sumo_cmd, approach, red_duration, speed_limit):
    
    
    traci.start(sumo_cmd)

    lane_IDs = traci.lane.getIDList()
    set_lane_speed_limits(lane_IDs, speed_limit)

    traci.vehicle.subscribe('vehicle_0', (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_NEXT_TLS))
    traci.vehicle.subscribe('vehicle_1', (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_NEXT_TLS))
    set_red_light(red_duration)
    
    # Looping things
    step = 0
    approaching = False
    while traci.simulation.getMinExpectedNumber() != 0:
        traci.simulationStep()
        step += 1
        sub_results = traci.vehicle.getSubscriptionResults('vehicle_0')
        sub_results_1 = traci.vehicle.getSubscriptionResults('vehicle_1')
        green_light = red_duration < step + 1# This is true if light is green
    
        #print('step', step)
        #print(f'green_light = {green_light}')
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

    return get_timeloss_diff('sumo/two_roads/tripinfo.xml', 'vehicle_0', 'vehicle_1')
