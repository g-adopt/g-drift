import numpy
import h5py
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data"


def path_to_dataset(h5finame: str):
    """_summary_

    Args:
        h5finame (str): filename

    Returns:
        str: path to the file
    """
    return DATA_PATH / h5finame


def load_dataset(dataset_name: str, table_names: list(str) = []):
    """_summary_

    Args:
        dataset_name (str): Filename
        table_names (list, optional): _description_. Defaults to [].

    Returns:
        dict: dictionary with all the datasets
    """
    dataset = {}
    # for cKDTree routines
    with h5py.File(path_to_dataset(dataset_name + '.h5'), 'r') as fi:
        keys_to_get = table_names if table_names else fi.keys()
        for key in keys_to_get:
            dataset[key] = numpy.array(fi.get(key))

    return dataset


def create_dataset_file(file_name: str, data_info: dict, metadata: dict):
    """
    Create an HDF5 file containing multiple 1D profiles, each with a name, and include metadata.

    Args:
        file_name (str): The name of the HDF5 file to create.
        data_info (dict): A dictionary where keys are profile names and values are numpy arrays representing the profiles.
        metadata (dict): A dictionary containing metadata about the data source.

    """
    # Create a new HDF5 file
    with h5py.File(DATA_PATH / file_name, 'w') as file:
        # Create a group for profiles

        # Add data profiles to the group
        for profile_name, data in data_info.items():
            # Each dataset is named after the profile name and contains the corresponding data
            file.create_dataset(profile_name, data=data)

        # Add metadata
        for key, value in metadata.items():
            file.attrs[key] = value
