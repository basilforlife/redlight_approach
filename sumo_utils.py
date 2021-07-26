import pickle
import xml.etree.ElementTree as ET

from Red_Light_Approach.approach import Approach



# ---------------------Args Processing Functions-------------------------------

# This class creates a context manager for editing an xml tree and writing result to new file
class XMLFile:
    def __init__(self, in_filename, out_filename=None):
        self.out_filename = out_filename
        self.tree = ET.parse(in_filename)
        self.root = self.tree.getroot()

    def __enter__(self):
        return self.root

    def __exit__(self, type, value, traceback):
        if self.out_filename:
            self.tree.write(self.out_filename)

def set_depart_pos_xml(root, x_min, edge_len):
    vehicles = root.findall('vehicle')
    for vehicle in vehicles: 
        vehicle.attrib['departPos'] = str(edge_len + x_min) # x_min is negative, so set vehicle abs(x_min) back from end of edge 

# This fn rewrites the rou.xml file to set accel and decel values
def set_accel_decel_xml(root, accel, decel):
    vtype = root.find('vType')
    vtype.attrib['accel'] = str(accel) 
    vtype.attrib['decel'] = str(decel) 
    vtype.attrib['emergencyDecel'] = str(decel) 

# This does the whole rou.xml processing
# Pass first edge_len that is for approaching the light	
def edit_rou_xml_file(in_filename, out_filename, approach, edge_len):
    with XMLFile(in_filename, out_filename) as xmlroot:
         set_depart_pos_xml(xmlroot, approach.x_min, edge_len)
         set_accel_decel_xml(xmlroot, approach.a_max, approach.a_max)

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

    # Check for incompatible flags
    if args.unpickle and args.approach_config_file:
        raise ValueError('Can only load approach object from config file or pickle')
    if args.unpickle:
        approach = read_pickle(args.unpickle)
    else:
        assert args.approach_config_file, 'Must include approach config file in flags'
        # Set up approach stuff
        approach = Approach(args.approach_config_file)
        approach.build_adjacency_matrix()
        approach.backward_pass()
    if args.pickle:
        write_pickle(args.pickle, approach)
    return approach
# --------------------/Args Processing Functions-------------------------------
