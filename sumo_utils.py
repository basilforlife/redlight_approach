import os
import sys
import xml.etree.ElementTree as ET


def add_sumo_path():

    # Ensure sumo is on the python load path
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

# Do xml parsing to get timeloss results
# Positive number represents approach having a beneficial effect
def get_timeloss_diff(filepath, vehicle_ID_0, vehicle_ID_1):
    timeloss_dict = get_timeloss_dict(filepath)
    timeloss_vehicle_0 = timeloss_dict[vehicle_ID_0]
    timeloss_vehicle_1 = timeloss_dict[vehicle_ID_1]
    timeloss_diff = timeloss_vehicle_1 - timeloss_vehicle_0
    return timeloss_diff

# Get a dict with vehicleID:timeLoss key value pairs from given tripinfo xml file
def get_timeloss_dict(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    timeloss_dict = {}
    for child in root:
        timeloss_dict[child.attrib['id']] = float(child.attrib['timeLoss'])
    return timeloss_dict

# This fn puts a vehicle in the left lane permanently
def set_lane_change_static(vehicle_ID):
    traci.vehicle.setLaneChangeMode(vehicle_ID, 0x000000000000)
