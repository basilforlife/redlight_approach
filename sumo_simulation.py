import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List

import traci
import traci.constants as tc

from redlight_approach.approach import Approach
from redlight_approach.state import State
from redlight_approach.sumo_utils import XMLFile


class SumoSimulation:
    """Represents a simulation in sumo and configures and runs that simulation

    This class manages a sumo simulation and modifies vehicle behavior according to an
    Approach object. It uses TraCI to interact with the simulation. It builds the simulation
    and matches the configuration to the Approach object configuration. Once configured, the
    simulation can be run any number of times with a different red light duration using the
    self.run(red_duration) method.
    """

    def __init__(
        self,
        approach: Approach,
        sumocfg_filename: str,
        first_edge_len: float,
        gui: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize and configure sumo simulation

        Parameters
        ----------
        approach
            Approach object to configure from and use for controlling approach
        sumocfg_filename
            Path to *.sumocfg file to run in sumo
        first_edge_len
            Length of first edge of route [m]
        gui
            Option to visualize simulations using sumo-gui. Passing True will enable
        verbose
            Option to print logs at each step of the simulation
        """
        self.add_sumo_path()
        self.approach = approach
        self.verbose = verbose
        route_filename = self.make_route_filename(sumocfg_filename)
        self.set_temp_route_filename(route_filename)
        self.set_sumo_command(gui, sumocfg_filename, self.temp_route_filename)
        self.edit_rou_xml_file(
            route_filename,
            self.temp_route_filename,
            self.approach.x_min,
            self.approach.a_max,
            first_edge_len,
        )

    def __del__(self) -> None:
        """Removes temporary files created during initialization"""
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
        return ["--configuration-file", sumocfg_filename]

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
        return ["--route-files", route_filename]

    def fcd_flag(self, fcd_filename: str) -> List[str]:
        """Returns a list of words to set an fcd file output

        Returns a list of command line options that tells sumo to write an
        fcd (floating car data) file to the given filepath. It contains information
        about the vehicles like position and velocity at each timestep.

        Parameters
        ----------
        fcd_filename : str
            The path to a *.xml file to log the fcd data

        Returns
        -------
        List[str]:
            A list of words to add to the command line call of sumo
        """
        return ["--fcd-output", fcd_filename]

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
        fcd_filename = os.path.join(os.path.dirname(sumocfg_filename), "fcd.xml")
        fcd_flag = self.fcd_flag(fcd_filename)
        self.sumo_command = (
            command + sumocfg_flag + steplength_flag + routefile_flag + fcd_flag
        )

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

    def make_route_filename(self, sumocfg_filename: str) -> str:
        """Returns a route filename that matches the provided sumocfg filename

        Parameters
        ----------
        sumocfg_filename
            Path of *.sumocfg file

        Returns
        ------
        str
            Path of *.rou.xml file

        Notes
        -----
        Fails if the file does not exist.
        """
        basename = os.path.basename(sumocfg_filename)
        name, extension = basename.split(".", 1)  # Only split on first period
        route_basename = name + ".rou.xml"
        route_filename = os.path.join(os.path.dirname(sumocfg_filename), route_basename)
        assert os.path.exists(route_filename)
        return route_filename

    def set_temp_route_filename(self, route_filename: str) -> None:
        """Sets a temporary name for the modified route file

        Parameters
        ----------
        route_filename
            Path of existing *.rou.xml file

        Notes
        -----
        Fails if the file already exists. If a SumoSimulation instance is initialized with the
        same config files as an instance that already exists, it will fail.
        """
        basename = os.path.basename(route_filename)
        name, extension = basename.split(".", 1)  # Only split on first period
        temp_basename = name + "_modified" + "." + extension
        temp_route_filename = os.path.join(
            os.path.dirname(route_filename), temp_basename
        )
        assert not os.path.exists(
            temp_route_filename
        )  # Fail before setting so __del__ doesn't touch
        self.temp_route_filename = temp_route_filename

    def get_timeloss_dict(self, filename: str) -> Dict[str, float]:
        """Make a dictionary mapping vehicle ID to time lost

        Returns a dictionary that maps a vehicle ID string to the time loss as
        calculated by sumo, which is the amount of time lost spent driving slower
        than the speed limit.

        Parameters
        ----------
        filename
            Path to *.xml file where tripinfo is logged

        Returns
        -------
        Dict[str, float]
            Dictionary with vehicleID:timeLoss key value pairs
        """
        timeloss_dict = {}
        with XMLFile(filename) as root:
            for child in root:
                timeloss_dict[child.attrib["id"]] = float(child.attrib["timeLoss"])
        return timeloss_dict

    def get_timeloss_diff(
        self, filename: str, vehicle_ID_0: str, vehicle_ID_1: str
    ) -> float:
        """Returns the difference in timeloss between the two specified vehicles

        Parameters
        ----------
        filename
            Path to *.xml file where tripinfo is written
        vehicle_ID_0
            ID string of vehicle
        vehicle_ID_1
            ID string of vehicle

        Returns
        -------
        float
            Difference in timeloss between the vehicles [s].

        Notes
        -----
        A positive result would indicate that vehicle 0 had a smaller timeloss
        than vehicle 1
        """
        timeloss_dict = self.get_timeloss_dict(filename)
        return timeloss_dict[vehicle_ID_1] - timeloss_dict[vehicle_ID_0]

    def print_vehicle_subscription_info(
        self, vehicle_ID: str, sub_results: dict
    ) -> None:
        """Prints info about a vehicle's state from sumo TraCI

        Parameters
        ----------
        vehicle_ID
            The vehicle's ID
        sub_results
            a TraCI subscription containing position and velocity info

        """
        if sub_results:
            pos = sub_results[86]
            vel = sub_results[64]
            print(f"{vehicle_ID}: pos = {pos:.2f}; vel = {vel:.2f}")
        else:
            print(f"{vehicle_ID} has left the route")

    def run(self, red_duration: float) -> float:
        """Runs a sumo simulation with the given red light duration

        This method runs the sumo simulation as configured by __init__().
        Runs are identical other than the specified red light duration. It returns the
        timeloss difference between the naive vehicle and the approach controlled vehicle.
        As it stands, this method runs the scenario where two vehicles approach a traffic
        light on their own one lane straight road.

        Parameters
        ----------
        red_duration
           Duration of red light [s]

        Returns
        -------
        float
            Difference in timeloss between the two approach methods
        """
        traci.start(self.sumo_command)
        self.set_speed_limit(self.approach.v_max)
        vehicle_ID_0 = "vehicle_0"
        vehicle_ID_1 = "vehicle_1"
        trl_distribution = self.approach.green_distribution.distribution
        subscriptions_tuple = (
            tc.VAR_ROAD_ID,
            tc.VAR_LANEPOSITION,
            tc.VAR_SPEED,
            tc.VAR_NEXT_TLS,
        )
        traci.vehicle.subscribe(vehicle_ID_0, subscriptions_tuple)
        traci.vehicle.subscribe(vehicle_ID_1, subscriptions_tuple)

        # Add a t_step here because sumo tells vehicles to speed up by the time the
        # light turns green, rather than allowing them to speed up after
        self.set_red_light(
            red_duration + self.approach.t_step, "0"
        )  # Traffic light ID = '0'

        # Looping things
        step = 0
        approaching = False
        approach_timestep = None
        while traci.simulation.getMinExpectedNumber() != 0:
            traci.simulationStep()
            step += 1

            sub_results_0 = traci.vehicle.getSubscriptionResults(vehicle_ID_0)
            sub_results_1 = traci.vehicle.getSubscriptionResults(vehicle_ID_1)
            green_light = (
                red_duration / self.approach.t_step < step + 1
            )  # This is true if light is green

            if self.verbose:
                time = traci.simulation.getTime()
                print(f"\n    STEP {step}: time = {time}")
                if step < len(trl_distribution):
                    print(f"Traffic light probability: {trl_distribution[step]:.2f}")
                if green_light:
                    print("LIGHT IS GREEN")
                self.print_vehicle_subscription_info(vehicle_ID_0, sub_results_0)
                self.print_vehicle_subscription_info(vehicle_ID_1, sub_results_1)

            # Check to see if vehicle_0 has a traffic light ahead, else continue
            # Everything below here in the while loop is approach control
            try:
                next_TLS = sub_results_0[112][0]
            except (KeyError, IndexError):
                continue

            # Statements in this block run if there is a traffic light ahead
            # -----------------BEGIN TLS-----------------------------------

            # Extract state from subscription results
            state = State(next_TLS[2] * -1, sub_results_0[64])

            # Begin approach
            # This runs only once when vehicle arrives in state space bounds
            if state.x >= self.approach.x_min and not green_light and not approaching:
                approach_timestep = 0
                approaching = True
                traci.vehicle.setColor(
                    vehicle_ID_0, (246, 186, 34)
                )  # Change color when approach starts

            # End approach
            # This runs only once to end the approach control
            if green_light and approaching:
                approaching = False
                traci.vehicle.setSpeed(vehicle_ID_0, -1)  # Hand control back to sumo
                traci.vehicle.setColor(
                    vehicle_ID_0, (42, 118, 189)
                )  # Change color when approach ends

            # This runs every timestep to control the approach
            if approaching:
                if self.verbose:
                    print(f"approach_timestep {approach_timestep}")
                next_state, _ = self.approach.forward_step(state, approach_timestep)
                traci.vehicle.setSpeed(vehicle_ID_0, next_state.v)
                approach_timestep += 1

            # -------------------END TLS-----------------------------------

        # Exit traci context
        if self.verbose:
            print("----------SIMULATION COMPLETE----------")
        traci.close()

        return self.get_timeloss_diff(
            "sumo/two_roads/tripinfo.xml", vehicle_ID_0, vehicle_ID_1
        )
