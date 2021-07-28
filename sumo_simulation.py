import os
import sys

import traci
import traci.constants as tc

from Red_Light_Approach.state import State
from Red_Light_Approach.sumo_utils import XMLFile


# This class can run sumo a bunch
class SumoSimulation():

    def __init__(self,
                 approach,
                 sumocfg_filename,
                 route_filename,
                 temp_route_filename,
                 first_edge_len,
                 gui=False, 
                 verbose=False):
        self.add_sumo_path()
        self.approach = approach
        self.verbose = verbose
        self.temp_route_filename = temp_route_filename
        self.set_sumo_command(gui, sumocfg_filename, temp_route_filename)
        self.edit_rou_xml_file(route_filename,
                               self.temp_route_filename,
                               self.approach.x_min,
                               self.approach.a_max,
                               first_edge_len)

    # This cleans up the object
    def __del__(self):
        os.remove(self.temp_route_filename)

    # This fn makes sure sumo is on the system path
    def add_sumo_path(self):

        # Ensure sumo is on the python load path
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

    # This fn puts a vehicle in the left lane permanently
    def set_lane_change_static(self, vehicle_ID):
        traci.vehicle.setLaneChangeMode(vehicle_ID, 0x000000000000)
    
    # This fn set the speed limit for all lanes in the simulation
    def set_lane_speed_limits(self, lane_IDs, speed_limit):
        for ID in lane_IDs:
            traci.lane.setMaxSpeed(ID, speed_limit)

    # This fn sets speed limit for whole simulation
    def set_speed_limit(self, speed_limit):
        lane_IDs = traci.lane.getIDList()
        self.set_lane_speed_limits(lane_IDs, speed_limit)
    
    # This fn takes a time and sets the light red for that long
    def set_red_light(self, red_duration, trafficlight_ID):
        traci.trafficlight.setPhase(trafficlight_ID, 2) # set traffic light to red 
        traci.trafficlight.setPhaseDuration(trafficlight_ID, red_duration)

    # This fn takes command line args and returns appropriate sumo config flags
    def sumocfg_flag(self, sumocfg_filename):
        return ['-c', sumocfg_filename]
    
    # This fn takes an approach object and returns matching sumo config flags
    def steplength_flag(self, t_step):
        return ['--step-length', str(t_step)]
    
    # This fn takes a fixed set of filenames to add to sumo flags
    def routefile_flag(self, route_filename):
        return ['-r', route_filename]
    
    # This fn takes the command line args and returns the sumo command
    def set_sumo_command(self, gui, sumocfg_filename, route_filename):
        if gui:
            sumo_location = '/usr/local/bin/sumo-gui' # Use graphical simulator
        else:
            sumo_location = '/usr/local/bin/sumo' # Use command line only simulator
        command = [sumo_location]
        sumocfg_flag = self.sumocfg_flag(sumocfg_filename)
        steplength_flag = self.steplength_flag(self.approach.t_step)
        routefile_flag = self.routefile_flag(route_filename)
        self.sumo_command = command + sumocfg_flag + steplength_flag + routefile_flag

    def set_depart_pos_xml(self, root, x_min, edge_len):
        vehicles = root.findall('vehicle')
        for vehicle in vehicles:
            vehicle.attrib['departPos'] = str(edge_len + x_min) # x_min is negative, so set vehicle abs(x_min) back from end of edge
    
    # This fn rewrites the rou.xml file to set accel and decel values
    def set_accel_decel_xml(self, root, accel, decel):
        vtype = root.find('vType')
        vtype.attrib['accel'] = str(accel)
        vtype.attrib['decel'] = str(decel)
        vtype.attrib['emergencyDecel'] = str(decel)
    
    # This does the whole rou.xml processing
    # Pass first edge_len that is for approaching the light
    def edit_rou_xml_file(self, in_filename, out_filename, x_min, a_max, edge_len):
        with XMLFile(in_filename, out_filename) as xmlroot:
             self.set_depart_pos_xml(xmlroot, x_min, edge_len)
             self.set_accel_decel_xml(xmlroot, a_max, a_max)
    
    # Get a dict with vehicleID:timeLoss key value pairs from given tripinfo xml file
    def get_timeloss_dict(self, filename):
        timeloss_dict = {}
        with XMLFile(filename) as root:
            for child in root:
                timeloss_dict[child.attrib['id']] = float(child.attrib['timeLoss'])
        return timeloss_dict
    
    # Do xml parsing to get timeloss results
    # Positive number represents approach having a beneficial effect
    def get_timeloss_diff(self, filename, vehicle_ID_0, vehicle_ID_1):
        timeloss_dict = self.get_timeloss_dict(filename)
        return timeloss_dict[vehicle_ID_1] - timeloss_dict[vehicle_ID_0]

    # This fn runs a sumo/traci simulation and returns the timeLoss difference
    def run(self, red_duration):
        traci.start(self.sumo_command)
        self.set_speed_limit(self.approach.v_max)
        traci.vehicle.subscribe('vehicle_0', (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_NEXT_TLS))
        traci.vehicle.subscribe('vehicle_1', (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_NEXT_TLS))
        self.set_red_light(red_duration, '0') # Traffic light ID = '0'
        
        # Looping things
        step = 0
        approaching = False
        while traci.simulation.getMinExpectedNumber() != 0:
            traci.simulationStep()
            step += 1
            sub_results = traci.vehicle.getSubscriptionResults('vehicle_0')
            sub_results_1 = traci.vehicle.getSubscriptionResults('vehicle_1')
            green_light = red_duration / self.approach.t_step < step + 1# This is true if light is green
        
            if self.verbose:
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
            if state.x >= self.approach.x_min and not green_light and not approaching:
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
                next_state, _ = self.approach.forward_step(state, approach_timestep)
                traci.vehicle.setSpeed('vehicle_0', next_state.v)
                if self.verbose: print(f'approach timestep = {approach_timestep}')
                approach_timestep += 1
        
            # -------------------END TLS-----------------------------------
        
        # Exit traci context
        traci.close()
    
        return self.get_timeloss_diff('sumo/two_roads/tripinfo.xml', 'vehicle_0', 'vehicle_1')
