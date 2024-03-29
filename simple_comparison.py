import argparse

import matplotlib.pyplot as plt
import numpy as np

from redlight_approach.sumo_simulation import SumoSimulation
from redlight_approach.sumo_utils import load_approach

if __name__ == "__main__":
    # Set up argparse
    parser = argparse.ArgumentParser(
        description="Run a SUMO simulation with an approach controller"
    )
    parser.add_argument(
        "-s",
        "--sumocfg_file",
        default="sumo/two_roads/f.sumocfg",
        type=str,
        help="path to .sumocfg file to run. Default sumo/two_roads/f.sumocfg",
    )
    parser.add_argument(
        "-c",
        "--approach_config_file",
        type=str,
        help="path to .json file to configure approach",
    )
    parser.add_argument(
        "-p", "--pickle", type=str, help="File to pickle approach object to"
    )
    parser.add_argument(
        "-u", "--unpickle", type=str, help="File to load approach object pickle from"
    )
    parser.add_argument(
        "-g",
        "--gui",
        action="store_true",
        help="Run the simulation in sumo-gui. Default is to run in sumo",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print information about each timestep to console",
    )
    parser.add_argument(
        "-N",
        default=1,
        type=int,
        help="This option runs the simulation N times and if N > 1, it plots them. Defaults to N=1.",
    )
    parser.add_argument(
        "-r",
        "--red-durations",
        nargs="+",
        type=float,
        help="This option overrides the random red light durations with the given durations. Overrides -N flag.",
    )
    args = parser.parse_args()

    # Load approach object according to command line args
    approach = load_approach(args)

    # Set first edge length as configured in sumo
    first_edge_len = 200

    # Configure SumoSimulation
    sumo_sim = SumoSimulation(
        approach,
        args.sumocfg_file,
        first_edge_len,
        args.gui,
        args.verbose,
    )

    # Get red_durations list
    if args.red_durations:
        red_durations = args.red_durations
    else:
        red_durations = approach.green_distribution.sample(
            args.N
        )  # random samples from distribution

    # Run simulation
    timeloss = [sumo_sim.run(red_duration) for red_duration in red_durations]

    # Plot, or log if only one data point
    if len(red_durations) > 1:
        plt.scatter(red_durations, timeloss)
        plt.title("test")
        plt.show()
    else:
        print(f"\nRed light duration: {red_durations[0]:.2f}s")

    # Print overall performance
    print(f"Mean time saved: {np.mean(timeloss):.2f}s")
