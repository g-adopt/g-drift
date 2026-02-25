"""Dataset download, caching, and integrity verification.

This module handles the lifecycle of dataset files: downloading from remote
S3-compatible storage (Digital Ocean Spaces), local caching in `gdrift/data/`,
SHA256 hash verification, and loading into memory as HDF5 datasets.

The I/O pipeline ensures:
1. **Download on demand**: Files are fetched only when first accessed
2. **Local caching**: Downloaded files persist in `gdrift/data/` for reuse
3. **Hash verification**: Every load checks SHA256 against the manifest
4. **Automatic retry**: Corrupted files are re-downloaded automatically
5. **Fallback**: boto3 S3 client with HTTPS/CDN fallback when boto3 unavailable

Storage Backend
---------------
Datasets are hosted on Digital Ocean Spaces (S3-compatible) with CDN acceleration:
- Bucket: `gadopt` (public read access)
- Prefix: `g-drift/`
- Endpoint: nyc3.digitaloceanspaces.com
- CDN: gadopt.nyc3.cdn.digitaloceanspaces.com

Files are stored with obfuscated names (SHA256 hashes of dataset names) to
prevent unauthorized scraping. The mapping is in `datasets.json`.

Key Functions
-------------
load_dataset : Main entry point - download, verify, and load HDF5 dataset
create_dataset_file : Developer utility for creating new datasets
download_all_datasets : Bulk download all registered datasets
path_to_dataset : Get local cache path for a dataset file
file_hash : Compute SHA256 hash of a file (for verification)

Internal Functions
------------------
_download_via_boto3 : S3 download using boto3 client (preferred)
_download_via_https : CDN download using urllib (fallback)
_verify_hash : Check file integrity against manifest hash
_get_s3_config : S3 configuration with environment variable overrides

Examples
--------
>>> import gdrift
>>> # Load dataset (downloads if not cached)
>>> with gdrift.load_dataset("1d_prem") as f:
...     density = f["density"][:]
>>>
>>> # Download all datasets for offline use
>>> gdrift.download_all_datasets()
>>>
>>> # Create a new dataset file (developer utility)
>>> import h5py
>>> with h5py.File("new_model.h5", "w") as f:
...     f.create_dataset("vs", data=vs_array)
>>> # Convert to gdrift dataset format
>>> gdrift.create_dataset_file("new_model.h5", "my_new_model")

Notes
-----
- Requires boto3 for optimal performance (pip install boto3)
- Falls back to HTTPS downloads if boto3 is not available
- Environment variables for testing:
  - GDRIFT_S3_ENDPOINT: Override S3 endpoint URL
  - GDRIFT_S3_BUCKET: Override bucket name
  - GDRIFT_S3_PREFIX: Override key prefix
- Hash mismatches trigger automatic re-download with warning
- Local cache location: `<gdrift_install>/gdrift/data/`

See Also
--------
gdrift.datasetnames : Dataset registry and manifest management
"""

import os
import warnings
import numpy
import h5py
from pathlib import Path
import hashlib
from .datasetnames import DATASET_REGISTRY, get_dataset_hash, get_manifest_config, hash_name


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


def _download_via_boto3(s3_config, h5finame, destination, display_name=None):
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
    desc = display_name or h5finame

    # Get file size for progress bar
    try:
        from tqdm import tqdm
        head = client.head_object(Bucket=s3_config["bucket"], Key=s3_key)
        total = head["ContentLength"]
        with tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
            client.download_file(
                s3_config["bucket"],
                s3_key,
                str(destination),
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )
    except ImportError:
        client.download_file(s3_config["bucket"], s3_key, str(destination))


def _download_via_https(cdn_url, h5finame, destination, display_name=None):
    """Fallback download using urllib when boto3 is not available."""
    import urllib.request
    url = cdn_url + h5finame
    desc = display_name or h5finame

    try:
        from tqdm import tqdm
        response = urllib.request.urlopen(url)
        total = int(response.headers.get("Content-Length", 0))
        with open(destination, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as pbar:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    except ImportError:
        urllib.request.urlretrieve(url, destination)


def download_dataset(h5finame: str, display_name=None):
    """Download a dataset from S3-compatible storage (boto3) or via HTTPS fallback."""
    destination = path_to_dataset(h5finame)
    s3_config = _get_s3_config()

    try:
        _download_via_boto3(s3_config, h5finame, destination, display_name=display_name)
    except ImportError:
        cdn_url = s3_config["cdn_url"]
        if not cdn_url:
            raise RuntimeError(
                "boto3 is not installed and no CDN URL is configured. "
                "Install boto3 (`pip install boto3`) or set a CDN URL in datasets.json."
            )
        _download_via_https(cdn_url, h5finame, destination, display_name=display_name)
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
        download_dataset(filepath.name, display_name=dataset_name)
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

    h5finame = hash_name(dataset_name) + ".h5"
    dataset_path = path_to_dataset(h5finame)

    if not dataset_path.exists():
        download_dataset(h5finame, display_name=dataset_name)

    _verify_and_maybe_redownload(dataset_name, dataset_path)

    with h5py.File(dataset_path, "r") as fi:
        # Detect structure: grouped (thermodynamic models) or flat (other datasets)
        if 'prop' in fi and isinstance(fi['prop'], h5py.Group):
            # GROUPED STRUCTURE (SLB thermodynamic models)
            # Flatten: /prop/rho → dataset['prop']['rho']
            for group_name in ['prop', 'opti']:
                if group_name in fi:
                    dataset[group_name] = {}
                    group_keys = table_names if table_names else fi[group_name].keys()
                    for key in group_keys:
                        if key in fi[group_name]:
                            dataset[group_name][key] = numpy.array(fi[group_name][key])
        else:
            # FLAT STRUCTURE (reference models, seismic models, solidus profiles)
            keys_to_get = table_names if table_names else fi.keys()
            for key in keys_to_get:
                dataset[key] = numpy.array(fi.get(key))

        # Load metadata (always at file level)
        for meta_key in fi.attrs.keys():
            metadata[meta_key] = fi.attrs[meta_key]

    if return_metadata:
        return dataset, metadata
    else:
        return dataset


def download_all_datasets(datasets=None):
    """Download registered datasets for offline use.

    Skips datasets already cached locally. Verifies hashes after download.

    Args:
        datasets (list of str, optional): Dataset names to download.
            If None, downloads every registered dataset.
    """
    if datasets is None:
        all_datasets = DATASET_REGISTRY.list_datasets()
    else:
        all_datasets = []
        for name in datasets:
            if name not in DATASET_REGISTRY:
                raise ValueError(
                    f"Unknown dataset '{name}'. "
                    f"Use DATASET_REGISTRY.list_datasets() to see available datasets."
                )
            all_datasets.append(DATASET_REGISTRY[name])
    total = len(all_datasets)
    cached = 0
    downloaded = 0
    failed = []

    for i, ds in enumerate(all_datasets, 1):
        h5finame = hash_name(ds.name) + ".h5"
        dataset_path = path_to_dataset(h5finame)

        if dataset_path.exists():
            cached += 1
            print(f"[{i}/{total}] {ds.name} — already cached")
            continue

        print(f"[{i}/{total}] Downloading {ds.name}...")
        try:
            download_dataset(h5finame, display_name=ds.name)
            _verify_and_maybe_redownload(ds.name, dataset_path)
            downloaded += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(ds.name)

    print(f"\nDone: {downloaded} downloaded, {cached} already cached, {len(failed)} failed.")
    if failed:
        print(f"Failed datasets: {', '.join(failed)}")


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
    """Compute cryptographic hash of a file.

    Reads the file in chunks to handle large files efficiently. Used for
    verifying dataset integrity against the manifest hashes.

    Parameters
    ----------
    path : str or Path
        Path to the file to hash.
    algo : str, optional
        Hash algorithm name (e.g., "sha256", "md5", "sha1"). Must be
        supported by hashlib. Default is "sha256".

    Returns
    -------
    str
        Hash digest in the format "algorithm:hexdigest", e.g.,
        "sha256:a1b2c3...". This format matches the manifest entries.

    Examples
    --------
    >>> from gdrift.io import file_hash, path_to_dataset
    >>> dataset_path = path_to_dataset("test_file.h5")
    >>> hash_str = file_hash(dataset_path)
    >>> print(hash_str)
    sha256:a1b2c3d4...

    Notes
    -----
    Reads file in 8192-byte chunks for memory efficiency. Suitable for
    files of any size.
    """
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"{algo}:{h.hexdigest()}"
