import argparse
import pickle
import xml.etree.ElementTree as ET
from typing import Any, Optional

from Red_Light_Approach.approach import Approach


class XMLFile:
    """Creates a context manager for reading/editing an xml tree from a file"""

    def __init__(self, in_filename: str, out_filename: Optional[str] = None) -> None:
        """Initialize with filenames

        Initializes by reading from in_filename and assigning the root of the xml
        object to `self.root`. If `out_filename` is included, the modified xml
        object will be written to that path.

        Parameters
        ----------
        in_filename
            Path to *.xml file to read
        out_filename
            Optional path to *.xml file to write to. Will not write changes if not included
        """
        self.out_filename = out_filename
        self.tree = ET.parse(in_filename)
        self.root = self.tree.getroot()

    def __enter__(self):
        """Enter context manager with root"""
        return self.root

    def __exit__(self, type, value, traceback):
        """Write changes to `out_filename` if provided"""
        if self.out_filename:
            self.tree.write(self.out_filename)


def read_pickle(filename: str) -> Any:
    """Read pickle from specified path

    Parameters
    ----------
    filename
        Path to pickle file

    Returns
    -------
    Any
        The object stored by the pickle file
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def write_pickle(filename: str, obj: Any) -> None:
    """Writes an object to a pickle file at the specified path

    Parameters
    ----------
    filename
        Path to write file to
    obj
        Object to write to file
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_approach(args: argparse.Namespace) -> Approach:
    """Load or create an Approach object according to command line args

    Parameters
    ----------
    args
        Commmand line args specifying how to make/load Approach object

    Returns
    -------
    Approach
        Configured approach object
    """

    # Check for incompatible flags
    if args.unpickle and args.approach_config_file:
        raise ValueError("Can only load approach object from config file or pickle")

    # Load from pickle file if specified
    if args.unpickle:
        approach = read_pickle(args.unpickle)

    # Create from configuration file
    else:
        assert args.approach_config_file, "Must include approach config file in flags"
        # Set up approach stuff
        approach = Approach(args.approach_config_file)
        approach.build_adjacency_matrix()
        approach.backward_pass()

    # Write to pickle file if specified
    if args.pickle:
        write_pickle(args.pickle, approach)
    return approach
