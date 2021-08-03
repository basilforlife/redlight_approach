import os
import sys
import xml.etree.ElementTree as ET
from typing import List

import traci
import traci.constants as tc

from Red_Light_Approach.state import State
from Red_Light_Approach.sumo_utils import XMLFile


# This class can run sumo a bunch
class SumoSimulation:
    def __init__(
        self,
        approach,
        sumocfg_filename,
        route_filename,
        temp_route_filename,
        first_edge_len,
        gui=False,
        verbose=False,
    ) -> None:
        self.add_sumo_path()
        self.approach = approach
        self.verbose = verbose
        self.temp_route_filename = temp_route_filename
        self.set_sumo_command(gui, sumocfg_filename, temp_route_filename)
        self.edit_rou_xml_file(
            route_filename,
            self.temp_route_filename,
            self.approach.x_min,
            self.approach.a_max,
            first_edge_len,
        )

    # This cleans up the object
    def __del__(self) -> None:
        os.remove(self.temp_route_filename)

    def add_sumo_path(self) -> None:
        """Add sumo to python load path, and exit if sumo path is not found"""
        if "SUMO_HOME" in os.environ:
            tools = os.path.join(os.environ["SUMO_HOME"], "tools")
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

    def set_lane_change_static(self, vehicle_ID: str) -> None:
        """Sets a vehicle's behavior to not change lanes

        Parameters
        ----------
        vehicle_ID
            String indicating vehicle to set behavior for
        """
        traci.vehicle.setLaneChangeMode(vehicle_ID, 0x000000000000)

    def set_lane_speed_limits(self, lane_IDs: List[str], speed_limit: float) -> None:
        """Sets speed limits for a list of lanes

        Parameters
        ----------
        lane_IDs
            List of lane_ID strings
        speed_limit
            Desired speed limit [m/s]
        """
        for ID in lane_IDs:
            traci.lane.setMaxSpeed(ID, speed_limit)

    def set_speed_limit(self, speed_limit: float) -> None:
        """Set speed limit for every lane in a simulation

        Parameters
        ----------
        speed_limit
            Desired speed limit [m/s]
        """
        lane_IDs = traci.lane.getIDList()
        self.set_lane_speed_limits(lane_IDs, speed_limit)

    def set_red_light(self, red_duration: float, trafficlight_ID: str) -> None:
        """Sets a traffic light to red for a given amount of time

        Parameters
        ----------
        red_duration
            Length of time for which light will be red [s]
        trafficlight_ID
            String indicating traffic light ID
        """
        traci.trafficlight.setPhase(trafficlight_ID, 2)  # set traffic light to red
        traci.trafficlight.setPhaseDuration(trafficlight_ID, red_duration)

    def sumocfg_flag(self, sumocfg_filename: str) -> List[str]:
        """Returns a list of words to specify a sumocfg file

        Parameters
        ----------
        sumocfg_filename
            The path to a *.sumocfg file to run in sumo

        Returns
        -------
        List[str]
            A list of words to add to the command line call of sumo
        """
        return ["-c", sumocfg_filename]

    def steplength_flag(self, t_step: float) -> List[str]:
        """Returns a list of words to specify a step length to sumo

        Parameters
        ----------
        t_step
            The length of one timestep in the sumo simulation [s]

        Returns
        -------
        List[str]
            A list of words to add to the command line call of sumo
        """
        return ["--step-length", str(t_step)]

    def routefile_flag(self, route_filename: str) -> List[str]:
        """Returns a list of words to specify a route filename to sumo

        Parameters
        ----------
        route_filename
            The path to a *.rou.xml file to run in sumo

        Returns
        -------
        List[str]
            A list of words to add to the command line call of sumo
        """
        return ["-r", route_filename]

    # This fn takes the command line args and returns the sumo command
    def set_sumo_command(
        self, gui: bool, sumocfg_filename: str, route_filename: str
    ) -> List[str]:
        """Assembles a command line call to run sumo with specific flags and options

        Parameters
        ----------
        gui
            Bool indicating if the gui should be used
        sumocfg_filename
            Path to *.sumocfg file to run in sumo
        route_filename
            Path to *.rou.xml file to use for simulation

        Returns
        -------
        List[str]
            List of words that make up the correct sumo command line call
        """
        if gui:
            sumo_location = "/usr/local/bin/sumo-gui"  # Use graphical simulator
        else:
            sumo_location = "/usr/local/bin/sumo"  # Use command line only simulator
        command = [sumo_location]
        sumocfg_flag = self.sumocfg_flag(sumocfg_filename)
        steplength_flag = self.steplength_flag(self.approach.t_step)
        routefile_flag = self.routefile_flag(route_filename)
        self.sumo_command = command + sumocfg_flag + steplength_flag + routefile_flag

    def set_depart_pos_xml(
        self, root: ET.Element, x_min: float, edge_len: float
    ) -> None:
        """Edit an xml ElementTree to make vehicles appear at a specified position

        Parameters
        ----------
        root
            The root Element of an xml ElementTree from a *.rou.xml file
        x_min
            The starting location on the first edge of the route [m]
        edge_len
            The length of the first edge of the route [m]

        Notes
        -----
        `x_min` is intended to match the x_min from the approach module, so
        it is a negative number representing the distance in meters from the
        traffic light intersection
        """
        vehicles = root.findall("vehicle")
        for vehicle in vehicles:
            vehicle.attrib["departPos"] = str(
                edge_len + x_min
            )  # x_min is negative, so set vehicle abs(x_min) back from end of edge

    def set_accel_decel_xml(self, root: ET.Element, accel: float, decel: float) -> None:
        """Edit an xml ElementTree to change vehicle accel and decel parameters

        Parameters
        ----------
        root
            The root Element of an xml ElementTree from a *.rou.xml file
        accel
            The maximum acceleration of the vehicle [m/(s^2)]
        decel
            The maximum deceleration of the vehicle [m/(s^2)]
        """
        vtype = root.find("vType")
        vtype.attrib["accel"] = str(accel)
        vtype.attrib["decel"] = str(decel)
        vtype.attrib["emergencyDecel"] = str(decel)

    def edit_rou_xml_file(
        self,
        in_filename: str,
        out_filename: str,
        x_min: float,
        a_max: float,
        edge_len: float,
    ) -> None:
        """Write a temporary *.rou.xml file with specified parameters changed

        Parameters
        ----------
        in_filename
            The path to the *.rou.xml file to edit
        out_filename
            The path to the temporary *.rou.xml file
        x_min
            The starting location on the first edge of the route [m]
        a_max
            The maximum acceleration of a vehicle [m/(s^2)]
        edge_len
            The length of the first edge of the route [m]
        """
        with XMLFile(in_filename, out_filename) as xmlroot:
            self.set_depart_pos_xml(xmlroot, x_min, edge_len)
            self.set_accel_decel_xml(xmlroot, a_max, a_max)

    # Get a dict with vehicleID:timeLoss key value pairs from given tripinfo xml file
    def get_timeloss_dict(self, filename):
        timeloss_dict = {}
        with XMLFile(filename) as root:
            for child in root:
                timeloss_dict[child.attrib["id"]] = float(child.attrib["timeLoss"])
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
        traci.vehicle.subscribe(
            "vehicle_0",
            (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_NEXT_TLS),
        )
        traci.vehicle.subscribe(
            "vehicle_1",
            (tc.VAR_ROAD_ID, tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_NEXT_TLS),
        )
        self.set_red_light(red_duration, "0")  # Traffic light ID = '0'

        # Looping things
        step = 0
        approaching = False
        while traci.simulation.getMinExpectedNumber() != 0:
            traci.simulationStep()
            step += 1
            sub_results = traci.vehicle.getSubscriptionResults("vehicle_0")
            sub_results_1 = traci.vehicle.getSubscriptionResults("vehicle_1")
            green_light = (
                red_duration / self.approach.t_step < step + 1
            )  # This is true if light is green

            if self.verbose:
                print("step", step)
                print(f"green_light = {green_light}")
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
                traci.vehicle.setColor(
                    "vehicle_0", (246, 186, 34)
                )  # Change color when approach starts

            # End approach
            # This runs only once to end the approach control
            if green_light and approaching:
                approaching = False
                traci.vehicle.setSpeed("vehicle_0", -1)  # Hand control back to sumo
                traci.vehicle.setColor(
                    "vehicle_0", (24, 255, 130)
                )  # Change color when approach ends

            # This runs every timestep to control the approach
            if approaching:
                next_state, _ = self.approach.forward_step(state, approach_timestep)
                traci.vehicle.setSpeed("vehicle_0", next_state.v)
                if self.verbose:
                    print(f"approach timestep = {approach_timestep}")
                approach_timestep += 1

            # -------------------END TLS-----------------------------------

        # Exit traci context
        traci.close()

        return self.get_timeloss_diff(
            "sumo/two_roads/tripinfo.xml", "vehicle_0", "vehicle_1"
        )
