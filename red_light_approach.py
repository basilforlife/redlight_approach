import argparse

import matplotlib.pyplot as plt
import numpy as np

from Red_Light_Approach.sumo_simulation import SumoSimulation
from Red_Light_Approach.sumo_utils import load_approach

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
args = parser.parse_args()

# Load approach object according to command line args
approach = load_approach(args)

# Configure some filenames
route_filename = "sumo/two_roads/f.rou.xml"
temp_route_filename = "sumo/two_roads/modified.rou.xml"
first_edge_len = 200

# Configure SumoSimulation
sumo_sim = SumoSimulation(
    approach,
    args.sumocfg_file,
    route_filename,
    temp_route_filename,
    first_edge_len,
    args.gui,
    args.verbose,
)

# Set up simulation stuff
num_samples = args.N
red_durations = approach.green_dist.sample(
    num_samples
)  # random sample from uniform distribution
timeloss = np.zeros_like(red_durations)

# Run simulation N times
for i, red_duration in enumerate(red_durations):
    timeloss[i] = sumo_sim.run(red_duration)

# Plot
if args.N > 1:
    plt.scatter(red_durations, timeloss)
    plt.title("test")
    plt.show()

# Print avg time saved
if args.N == 1:
    print(f"Red light duration: {red_durations[0]}")
print(f"Mean time saved: {np.mean(timeloss):.2f}s")
