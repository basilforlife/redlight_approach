import os
import pickle
from random import uniform
import sys
import xml.etree.ElementTree as ET

import traci

from Red_Light_Approach.approach import Approach

def add_sumo_path():

    # Ensure sumo is on the python load path
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

# This fn puts a vehicle in the left lane permanently
def set_lane_change_static(vehicle_ID):
    traci.vehicle.setLaneChangeMode(vehicle_ID, 0x000000000000)

# This fn takes a time and sets the light red for that long
def set_red_light(red_duration):
    traci.trafficlight.setPhase('0', 2) # set traffic light id='0' to red
    traci.trafficlight.setPhaseDuration('0', red_duration)

# ---------------------TimeLoss Parsing Functions------------------------------

# Get a dict with vehicleID:timeLoss key value pairs from given tripinfo xml file
def get_timeloss_dict(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    timeloss_dict = {}
    for child in root:
        timeloss_dict[child.attrib['id']] = float(child.attrib['timeLoss'])
    return timeloss_dict

# Do xml parsing to get timeloss results
# Positive number represents approach having a beneficial effect
def get_timeloss_diff(filepath, vehicle_ID_0, vehicle_ID_1):
    timeloss_dict = get_timeloss_dict(filepath)
    timeloss_vehicle_0 = timeloss_dict[vehicle_ID_0]
    timeloss_vehicle_1 = timeloss_dict[vehicle_ID_1]
    timeloss_diff = timeloss_vehicle_1 - timeloss_vehicle_0
    return timeloss_diff
# --------------------/TimeLoss Parsing Functions------------------------------


# ---------------------Args Processing Functions-------------------------------

# This fn takes the command line args and returns the sumo command
def sumo_command(args):
    if args.gui:
        sumo_location = '/usr/local/bin/sumo-gui' # Use graphical simulator
    else:
        sumo_location = '/usr/local/bin/sumo' # Use command line only simulator
    return [sumo_location, '-c', args.path_to_sumocfg]

# This fn takes a filename containing a pickled object and returns the object
def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# This fn takes an object and a filename and pickles the object in the file
def write_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

# This fn takes command line args and loads an approach object accordingly
def load_approach(args):
    if args.unpickle:
        approach = read_pickle(args.pickle_file)
    else:
        # Set up approach stuff
        approach = Approach(args.path_to_approach_config)
        approach.build_adjacency_matrix()
        approach.backward_pass()
    if args.pickle:
        write_pickle(args.pickle_file, approach)
    return approach
# --------------------/Args Processing Functions-------------------------------
