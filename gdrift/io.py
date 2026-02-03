import os
import warnings
import numpy
import h5py
from pathlib import Path
import hashlib
from .datasetnames import DATASET_REGISTRY, get_dataset_hash, get_manifest_config


DATA_PATH = Path(__file__).resolve().parent / "data"


def _get_s3_config():
    """Get S3 configuration, allowing env var overrides."""
    config = get_manifest_config()
    return {
        "endpoint_url": os.environ.get("GDRIFT_S3_ENDPOINT", config["endpoint_url"]),
        "bucket": os.environ.get("GDRIFT_S3_BUCKET", config["bucket"]),
        "prefix": os.environ.get("GDRIFT_S3_PREFIX", config["prefix"]),
        "cdn_url": config["cdn_url"],
    }


def path_to_dataset(h5finame: str):
    """Return the local path for a dataset file.

    Args:
        h5finame (str): filename

    Returns:
        Path: path to the file
    """
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    return DATA_PATH / h5finame


def _download_via_boto3(s3_config, h5finame, destination):
    """Download a file from S3-compatible storage using boto3."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=s3_config["endpoint_url"],
        config=Config(signature_version=UNSIGNED),
    )
    s3_key = s3_config["prefix"] + h5finame

    # Get file size for progress bar
    try:
        from tqdm import tqdm
        head = client.head_object(Bucket=s3_config["bucket"], Key=s3_key)
        total = head["ContentLength"]
        with tqdm(total=total, unit="B", unit_scale=True, desc=h5finame) as pbar:
            client.download_file(
                s3_config["bucket"],
                s3_key,
                str(destination),
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )
    except ImportError:
        client.download_file(s3_config["bucket"], s3_key, str(destination))


def _download_via_https(cdn_url, h5finame, destination):
    """Fallback download using urllib when boto3 is not available."""
    import urllib.request
    url = cdn_url + h5finame

    try:
        from tqdm import tqdm
        response = urllib.request.urlopen(url)
        total = int(response.headers.get("Content-Length", 0))
        with open(destination, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=h5finame) as pbar:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    except ImportError:
        urllib.request.urlretrieve(url, destination)


def download_dataset(h5finame: str):
    """Download a dataset from S3-compatible storage (boto3) or via HTTPS fallback."""
    destination = path_to_dataset(h5finame)
    s3_config = _get_s3_config()

    try:
        _download_via_boto3(s3_config, h5finame, destination)
    except ImportError:
        cdn_url = s3_config["cdn_url"]
        if not cdn_url:
            raise RuntimeError(
                "boto3 is not installed and no CDN URL is configured. "
                "Install boto3 (`pip install boto3`) or set a CDN URL in datasets.json."
            )
        _download_via_https(cdn_url, h5finame, destination)
    except Exception as e:
        raise FileNotFoundError(
            f"Dataset {h5finame} could not be downloaded. Error: {e}"
        )

    return destination


def _verify_hash(filepath, expected_hash):
    """Verify a file's SHA256 hash against the expected value.

    Returns True if hashes match, False otherwise.
    """
    if expected_hash is None:
        return True
    actual = file_hash(filepath)
    return actual == expected_hash


def _verify_and_maybe_redownload(dataset_name, filepath):
    """Verify hash of a local file; re-download if mismatch."""
    expected = get_dataset_hash(dataset_name)
    if expected is None:
        return

    if not _verify_hash(filepath, expected):
        warnings.warn(
            f"Hash mismatch for {dataset_name}. Expected {expected}, "
            f"got {file_hash(filepath)}. Re-downloading.",
            stacklevel=3,
        )
        download_dataset(filepath.name)
        if not _verify_hash(filepath, expected):
            raise RuntimeError(
                f"Hash verification failed for {dataset_name} even after re-download."
            )


def load_dataset(dataset_name: str, table_names=[], return_metadata=False):
    """Load a dataset from local cache, downloading if necessary.

    Args:
        dataset_name (str): Dataset name (without .h5 extension)
        table_names (list, optional): Specific tables to load. Defaults to [].
        return_metadata (bool, optional): Whether to return file-level metadata.

    Returns:
        dict: dictionary with all the datasets (and optionally metadata tuple)
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Use DATASET_REGISTRY.get_dataset_names() to see available datasets."
        )

    dataset = {}
    metadata = {}

    dataset_path = path_to_dataset(dataset_name + ".h5")

    if not dataset_path.exists():
        download_dataset(dataset_name + ".h5")

    _verify_and_maybe_redownload(dataset_name, dataset_path)

    with h5py.File(dataset_path, "r") as fi:
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
    with h5py.File(DATA_PATH / file_name, "w") as file:
        for profile_name, data in data_info.items():
            file.create_dataset(profile_name, data=data)

        for key, value in metadata.items():
            file.attrs[key] = value


def file_hash(path, algo="sha256"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"{algo}:{h.hexdigest()}"
