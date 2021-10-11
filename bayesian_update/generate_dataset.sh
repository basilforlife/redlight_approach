#!/bin/bash

sumo_path="../sumo/intersection"

duarouter -n $sumo_path/f.net.xml -r $sumo_path/f.rou.xml --randomize-flows -o $sumo_path/randomflows.rou.xml
echo "Routes randomized"

sumo -n $sumo_path/f.net.xml -r $sumo_path/randomflows.rou.xml --fcd-output $sumo_path/fcd.xml --fcd-output.attributes speed,lane,y --step-length 0.99
echo "Simulation complete"

python extract_data_from_fcd.py
echo "Dataset file written"

