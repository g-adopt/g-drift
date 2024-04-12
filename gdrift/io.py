import numpy
import h5py
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data"


def path_to_dataset(h5finame):
    return DATA_PATH / h5finame


def load_dataset(dataset_name, table_names=[]):
    dataset = {}
    # for cKDTree routines
    with h5py.File(path_to_dataset(dataset_name + '.hdf5'), 'r') as fi:
        keys_to_get = table_names if table_names else fi.keys()
        for key in keys_to_get:
            dataset[key] = numpy.array(fi.get(key))

    return dataset