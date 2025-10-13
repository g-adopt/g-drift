import numpy
import h5py
import pooch
from pathlib import Path
import hashlib
from .datasetnames import AVAILABLE_DATASETS, get_dataset_hash


DATA_PATH = Path(__file__).resolve().parent / "data"
BASE_URL = "https://data.gadopt.org/g-drift/"


def path_to_dataset(h5finame: str):
    """_summary_

    Args:
        h5finame (str): filename

    Returns:
        str: path to the file
    """
    # Making sure the data directory is generated
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    return DATA_PATH / h5finame


def download_dataset(h5finame: str):
    """Downloads the dataset using pooch with hash verification if available."""
    url = BASE_URL + h5finame

    # Get the dataset name (remove .h5 extension)
    dataset_name = h5finame.replace('.h5', '')

    # Get hash from dataset registry
    known_hash = get_dataset_hash(dataset_name)

    # Use Pooch to fetch the file with a progress bar and hash verification
    try:
        print(f"Downloading {h5finame} from {url}")
        if known_hash:
            print(f"Using hash verification: {known_hash}")
        file_path = pooch.retrieve(
            url=url,
            known_hash=known_hash,  # Hash verification for integrity
            path=DATA_PATH,
            fname=h5finame,
            progressbar=True
        )
        print(f"Downloaded {h5finame} successfully to {file_path}.")

        if known_hash:
            print("✓ Hash verification passed - file integrity confirmed.")
        else:
            print("⚠ No hash available for verification - consider adding hash to dataset registry.")
    except Exception as e:
        raise FileNotFoundError(
            f"Dataset {h5finame} not found on the server or could not be downloaded. Error: {e}")

    return Path(file_path)


def load_dataset(dataset_name: str, table_names=[], return_metadata=False):
    """_summary_

    Args:
        dataset_name (str): Filename
        table_names (list, optional): _description_. Defaults to [].

    Returns:
        dict: dictionary with all the datasets
    """
    dataset = {}
    metadata = {}

    # Full path to the dataset
    dataset_path = path_to_dataset(dataset_name + '.h5')

    if not dataset_path.exists():
        if str(dataset_name) in [key.name for key in AVAILABLE_DATASETS]:
            downloaded_dataset_path = download_dataset(dataset_name + ".h5")
            if downloaded_dataset_path != path_to_dataset(dataset_name + ".h5"):
                raise ValueError((
                    f"Expected to have the file as {dataset_path}, "
                    f"but is on {downloaded_dataset_path}")
                )
        else:
            raise FileNotFoundError(
                f"Dataset {dataset_name} is neither on disk or on our server!")

    # for cKDTree routines
    with h5py.File(path_to_dataset(dataset_name + '.h5'), 'r') as fi:
        keys_to_get = table_names if table_names else fi.keys()
        for key in keys_to_get:
            dataset[key] = numpy.array(fi.get(key))

        for meta_key in fi.attrs.keys():
            metadata[meta_key] = fi.attrs[meta_key]

    if return_metadata:
        return dataset, metadata
    else:
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


def file_hash(path, algo="sha256"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"{algo}:{h.hexdigest()}"
