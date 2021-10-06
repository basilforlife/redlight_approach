from redlight_approach.sumo_utils import XMLFile, write_pickle


# Initialize empty dataset
dataset = []


# Load data from fcd file
filename = "../sumo/intersection/fcd.xml"
with XMLFile(filename) as root:

    # Iterate through timesteps
    for timestep in root:
        time = timestep.attrib["time"]

        # Iterate through vehicles, cleaning those past the light
        for vehicle in timestep:
            data = vehicle.attrib
            if data["lane"] == "gneE5_0":
                dataset.append([time, data["speed"], data["y"]])

# Write result to pickle
write_pickle("dataset.pickle", dataset)
