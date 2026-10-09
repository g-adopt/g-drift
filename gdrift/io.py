"""Dataset download, caching, and integrity verification.

This module handles the lifecycle of dataset files: downloading them from the
G-ADOPT server, caching them on the local disk, checking their SHA256 hash
against the manifest, and loading them into memory from HDF5.

The I/O pipeline ensures:
1. **Download on demand**: Files are fetched only when first accessed
2. **Local caching**: Downloaded files persist in a user cache directory
3. **Hash verification**: Every load checks SHA256 against the manifest
4. **Automatic repair**: A cached file with the wrong hash is downloaded again
5. **Fallback**: If the CDN fails, the file is downloaded from the server itself

Storage Backend
---------------
Datasets are HDF5 files on DigitalOcean Spaces (S3-compatible, public read),
served through a CDN. The bucket, prefix, endpoint and CDN URL are in
`datasets.json`.

Each file on the server is named after the SHA256 of its content
(`<sha256>.h5`), and the manifest gives that name in the `filename` field of
each dataset. A file on the server is never overwritten. When a dataset is
fixed, the new content is uploaded under its new name, and the manifest of the
new gdrift release points to it. Older releases keep their manifest, so they
keep loading the content they were released with.

gdrift 0.1.3 and earlier named the files after the SHA256 of the dataset name
(`hash_name(name) + ".h5"`). Those files stay on the server unchanged, so these
releases keep working. They are also looked up in the old local cache
directory (see below).

Local Cache
-----------
Files are cached in `data_dir()`:
- `$GDRIFT_DATA_DIR`, if this environment variable is set. Use it on clusters
  to put the cache on a large shared file system, and to download on a login
  node before running on compute nodes without internet access.
- Otherwise the user cache directory of the platform, from `pooch.os_cache`
  (`~/Library/Caches/gdrift` on macOS, `~/.cache/gdrift` on Linux).

Before downloading, the loader also looks in `DATA_PATH` (`gdrift/data/` in the
installed package). gdrift 0.1.3 and earlier cached files there, and the
developer scripts in `scripts/` write new files there. A file in `DATA_PATH` is
used only if its hash matches the manifest.

Key Functions
-------------
load_dataset : Main entry point - download, verify, and load HDF5 dataset
fetch_dataset : Return the local path of a verified dataset file
create_dataset_file : Developer utility for creating new datasets
download_all_datasets : Bulk download all registered datasets
data_dir : Local cache directory
path_to_dataset : Local cache path for a dataset file
file_hash : Compute SHA256 hash of a file (for verification)

Examples
--------
>>> import gdrift
>>> # Load dataset (downloads if not cached)
>>> data = gdrift.load_dataset("1d_prem")
>>> density = data["density"]
>>>
>>> # Download all datasets for offline use
>>> gdrift.download_all_datasets()
>>> # Or only some of them
>>> gdrift.download_all_datasets(["1d_prem", "SLB_21_pyroliteCFMAS"])

Notes
-----
- Downloads use pooch over HTTPS, first from the CDN, then from the server
  itself.
- Environment variables:
  - GDRIFT_DATA_DIR: Local cache directory
  - GDRIFT_DATA_URL: Base URL to download from instead of the CDN and the
    server (for testing or a mirror). It must end with "/".
- A failed or corrupted download never replaces a cached file, because pooch
  checks the hash of a temporary file before it moves it into place.
- In a parallel run, a lock file next to the target (`<file>.lock`) makes one
  process download a missing file while the others wait. Downloading the
  datasets once before a large parallel run is still the most reliable way,
  for example with `download_all_datasets(["name", ...])` on a login node.
- Downloaded files get the permissions 0666 minus the umask, so a cache on a
  shared file system can be read by other users.
- The cache is never cleaned. Each version of a dataset has its own file, so
  old versions stay until the cache directory is deleted by hand.

See Also
--------
gdrift.datasetnames : Dataset registry and manifest management
"""

import contextlib
import logging
import os
import warnings
import numpy
import h5py
from pathlib import Path
import hashlib
from .datasetnames import DATASET_REGISTRY, get_manifest_config, hash_name


# Directory inside the installed package. gdrift 0.1.3 and earlier cached the
# downloaded files here (named `hash_name(name) + ".h5"`), and the developer
# scripts write newly built files here. The loader looks here before it
# downloads anything, so existing caches and freshly built files are reused.
DATA_PATH = Path(__file__).resolve().parent / "data"

# Environment variable that overrides the cache directory
DATA_DIR_ENV = "GDRIFT_DATA_DIR"

# Environment variable that overrides the download base URL
DATA_URL_ENV = "GDRIFT_DATA_URL"

# Number of extra download attempts for each URL. Pooch retries both
# connection errors and hash mismatches, and waits 1 s, 2 s, ... in between.
RETRY_IF_FAILED = 2

# Timeout in seconds for connecting and for each read of a download. Without
# it, a server that accepts the connection and never answers blocks for ever,
# and the next URL is never tried.
DOWNLOAD_TIMEOUT = 60

# Size in bytes of the chunks in which a download is written. Larger chunks
# than the pooch default (1 KiB) give a much higher throughput on fast links.
DOWNLOAD_CHUNK_SIZE = 1 << 20


def _import_pooch():
    """Import pooch on first use and return the module.

    pooch imports `tqdm.auto`. In a Jupyter kernel without ipywidgets, that
    import prints a TqdmWarning ("IProgress not found"). Importing pooch only
    when a download is needed keeps `import gdrift` free of it, and the
    warning is suppressed here because gdrift uses the plain text progress bar.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*IProgress not found.*")
        import pooch
    return pooch


def data_dir() -> Path:
    """Return the local cache directory for downloaded datasets.

    The environment variable is read at every call, so a change of
    `GDRIFT_DATA_DIR` takes effect without a restart. The directory is not
    created here; it is created when the first file is downloaded.

    Returns:
        Path: `$GDRIFT_DATA_DIR` if it is set, otherwise the user cache
        directory of the platform for "gdrift" (from `pooch.os_cache`).
    """
    env = os.environ.get(DATA_DIR_ENV)
    if env:
        return Path(env).expanduser()
    return Path(_import_pooch().os_cache("gdrift"))


def path_to_dataset(h5finame: str) -> Path:
    """Return the path of a dataset file in the cache directory.

    Creates the cache directory if it does not exist. The file itself is not
    downloaded or checked.

    Args:
        h5finame (str): File name, as given in the `filename` field of the manifest.

    Returns:
        Path: `data_dir() / h5finame`.
    """
    directory = data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory / h5finame


def _base_urls():
    """Return the base URLs to download from, in the order to try them.

    The CDN comes first because it is faster and caches the files close to
    the user. The server itself (path-style URL of the public bucket) is the
    fallback when the CDN fails. Both serve the same objects. Files on the
    server are never overwritten, so the CDN cannot serve outdated content for
    a file name.

    Returns:
        list of str: Base URLs that end with "/". A file is at base URL +
        file name.
    """
    # An explicit override replaces both sources (tests, mirrors)
    override = os.environ.get(DATA_URL_ENV)
    if override:
        return [override]

    config = get_manifest_config()
    urls = []
    if config["cdn_url"]:
        urls.append(config["cdn_url"])
    # Path-style URL of the public bucket, for example
    # https://syd1.digitaloceanspaces.com/gadopt/g-drift/
    urls.append(f"{config['endpoint_url'].rstrip('/')}/{config['bucket']}/{config['prefix']}")
    return urls


def _verify_hash(filepath, expected_hash):
    """Verify a file's SHA256 hash against the expected value.

    Args:
        filepath (str or Path): File to check.
        expected_hash (str or None): Hash in the form "sha256:<hex>", or None.

    Returns:
        bool: True if the hashes match or if `expected_hash` is None.
    """
    if expected_hash is None:
        return True
    return file_hash(filepath) == expected_hash


def _is_valid(path, expected_hash):
    """Return True if `path` exists, can be read, and has the expected hash.

    A file that cannot be read (for example a file of another user in a
    shared cache) counts as not valid, so the caller can look elsewhere or
    download it again.
    """
    try:
        return path.is_file() and _verify_hash(path, expected_hash)
    except OSError:
        return False


def _umask():
    """Return the current umask of the process.

    Python can read the umask only by setting it, so it is set and restored
    at once. The short change is not visible to other processes.
    """
    mask = os.umask(0)
    os.umask(mask)
    return mask


@contextlib.contextmanager
def _download_lock(target):
    """Hold an exclusive lock while one process downloads `target`.

    In a parallel run (for example G-ADOPT with MPI), every rank can ask for
    the same missing file at the same time. The lock makes one process
    download it while the others wait and then find it in the cache. The lock
    is `flock` on the file `<target>.lock` next to the target, which works
    across processes and, on Linux, across nodes on NFS.

    If locking is not possible (no `fcntl` on the platform, a lock file that
    cannot be opened, or a file system without `flock`, such as Lustre
    mounted without the flock option), the download goes ahead without a
    lock. That is still correct: each process downloads to its own temporary
    file and moves a checked file into place. It only costs extra transfers.

    Args:
        target (Path): The file that is about to be downloaded.
    """
    try:
        import fcntl
    except ImportError:
        # No flock on this platform (Windows)
        yield
        return

    lock_path = target.with_name(target.name + ".lock")
    try:
        handle = open(lock_path, "a")
    except OSError:
        # For example a lock file of another user in a shared cache
        yield
        return

    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            locked = True
        except OSError:
            # The file system does not support flock
            locked = False
        try:
            yield
        finally:
            if locked:
                fcntl.flock(handle, fcntl.LOCK_UN)


class _ProgressBar:
    """Progress bar for pooch that shows the dataset name.

    pooch accepts any object with a `total` attribute and `update`, `reset`
    and `close` methods, and tests it with `if progressbar:`. A tqdm bar
    without a total cannot be tested that way, so this wrapper is always true
    and creates the tqdm bar only when pooch sets the total. No bar appears
    for a URL that fails before the download starts.
    """

    def __init__(self, desc):
        """Store the description; the tqdm bar is created when the total is known."""
        self.desc = desc
        self._bar = None

    def __bool__(self):
        """Always true, so that pooch uses this progress bar."""
        return True

    @property
    def total(self):
        """Total number of bytes, or None before the download starts."""
        return self._bar.total if self._bar is not None else None

    @total.setter
    def total(self, value):
        """Create the plain text tqdm bar for a download of `value` bytes."""
        from tqdm import tqdm
        self._bar = tqdm(total=value, desc=self.desc, unit="B", unit_scale=True, ncols=79, leave=True)

    def update(self, n):
        """Advance the bar by `n` bytes."""
        if self._bar is not None:
            self._bar.update(n)

    def reset(self):
        """Set the bar back to zero (pooch then fills it to the total)."""
        if self._bar is not None:
            self._bar.reset()

    def close(self):
        """Close the bar."""
        if self._bar is not None:
            self._bar.close()


@contextlib.contextmanager
def _quiet_pooch(pooch):
    """Hide pooch's INFO messages during a download.

    pooch logs one line per download and per retry, with the hashed file name
    only, through its own logger that the normal logging configuration does
    not reach. The progress bar already names the dataset, and a failure is
    reported in the exception, so the INFO lines are hidden. Warnings and
    errors are still shown.
    """
    logger = pooch.get_logger()
    level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(level)


def _download(target, known_hash, display_name):
    """Download `target.name` into `target.parent` from each base URL in turn.

    Args:
        target (Path): Destination in the cache directory.
        known_hash (str or None): Expected hash, "sha256:<hex>".
        display_name (str): Name shown in the progress bar and in errors.

    Returns:
        Path: The downloaded file, with the expected hash.

    Raises:
        RuntimeError: If no URL gives the file with the expected hash. The
            last download error is chained as the cause.
        OSError: For local problems such as a full disk or missing write
            permission. These would fail for every URL, so they are raised
            at once.
    """
    pooch = _import_pooch()
    import requests  # a dependency of pooch

    errors = []
    last_error = None
    for base_url in _base_urls():
        fetcher = pooch.create(
            path=target.parent,
            base_url=base_url,
            registry={target.name: known_hash},
            retry_if_failed=RETRY_IF_FAILED,
        )
        # Plain text progress bar that names the dataset (the file name on
        # the server is only a hash)
        progress = _ProgressBar(display_name)
        downloader = pooch.HTTPDownloader(
            progressbar=progress, timeout=DOWNLOAD_TIMEOUT, chunk_size=DOWNLOAD_CHUNK_SIZE)
        try:
            with _quiet_pooch(pooch):
                path = Path(fetcher.fetch(target.name, downloader=downloader))
        except (ValueError, requests.exceptions.RequestException) as e:
            # ValueError: hash mismatch. RequestException: connection, HTTP
            # status or timeout errors. Try the next URL.
            progress.close()
            errors.append(f"{base_url}{target.name}: {e}")
            last_error = e
            continue

        # pooch writes a temporary file with mode 0600 and moves it into
        # place. Give the file the normal permissions (0666 minus the umask)
        # so that other users of a shared cache can read it.
        try:
            path.chmod(0o666 & ~_umask())
        except OSError:
            pass
        return path

    raise RuntimeError(
        f"Dataset {display_name} could not be downloaded with the expected hash "
        f"{known_hash}. Tried:\n  " + "\n  ".join(errors)
    ) from last_error


def _fetch(filename, known_hash, display_name, legacy_names=()):
    """Return the local path of a verified file, downloading it if needed.

    The lookup order is:

    1. The cache directory `data_dir()`.
    2. The package directory `DATA_PATH`, under the file name and under each
       of `legacy_names`. Such a file is used in place (not copied).
    3. A download into `data_dir()`, from each base URL of `_base_urls()` in
       turn, under a lock so that parallel processes download it only once.

    A local file is used only if its hash matches. Steps 1 and 2 need no
    network access and do not write to the cache directory, so a read-only
    or missing cache directory does not matter when a valid file is in
    `DATA_PATH`.

    Args:
        filename (str): File name on the server and in the cache.
        known_hash (str or None): Expected hash, "sha256:<hex>". With None, the
            file is not checked.
        display_name (str): Name used in messages, normally the dataset name.
        legacy_names (iterable of str): Other names under which the file can be
            in `DATA_PATH`, for example the name-based file name of gdrift
            0.1.3 and earlier.

    Returns:
        Path: Local path of a file whose hash matches `known_hash`.

    Raises:
        RuntimeError: If no source gives a file with the expected hash.
        OSError: If the cache directory cannot be created or written.
    """
    target = data_dir() / filename

    # 1. A valid file in the cache directory
    if _is_valid(target, known_hash):
        return target

    # 2. A valid file in the package directory, used in place
    for name in (filename, *legacy_names):
        candidate = DATA_PATH / name
        if _is_valid(candidate, known_hash):
            return candidate

    # 3. Download. Another process can have downloaded the file while this
    # one waited for the lock, so check again once the lock is held.
    target.parent.mkdir(parents=True, exist_ok=True)
    with _download_lock(target):
        if _is_valid(target, known_hash):
            return target
        return _download(target, known_hash, display_name)


def fetch_dataset(dataset_name: str) -> Path:
    """Return the local path of a verified dataset file, downloading it if needed.

    Args:
        dataset_name (str): Dataset name as in the manifest (without .h5).

    Returns:
        Path: Local HDF5 file whose SHA256 matches the manifest.

    Raises:
        ValueError: If the dataset is not in the registry.
        RuntimeError: If the file cannot be obtained with the expected hash.
        OSError: If the cache directory cannot be created or written.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Use DATASET_REGISTRY.get_dataset_names() to see available datasets."
        )
    ds = DATASET_REGISTRY.get_dataset(dataset_name)
    # The name-based file name lets the loader reuse caches of gdrift 0.1.3
    # and earlier and files written by the developer scripts.
    return _fetch(ds.get_filename(), ds.file_hash, ds.name, legacy_names=(hash_name(ds.name) + ".h5",))


def load_dataset(dataset_name: str, table_names=[], return_metadata=False):
    """Load a dataset from local cache, downloading if necessary.

    Args:
        dataset_name (str): Dataset name (without .h5 extension)
        table_names (list, optional): Specific tables to load. Defaults to [].
        return_metadata (bool, optional): Whether to return file-level metadata.

    Returns:
        dict: dictionary with all the datasets (and optionally metadata tuple)
    """
    # Local path of a file whose hash matches the manifest (downloads if needed)
    dataset_path = fetch_dataset(dataset_name)

    dataset = {}
    metadata = {}

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

    A dataset counts as cached if a local file with the right hash exists
    (in `data_dir()` or `DATA_PATH`). Every other dataset is downloaded and
    checked. Failures are collected and reported at the end, so one missing
    dataset does not stop the others.

    Args:
        datasets (list of str, optional): Dataset names to download.
            If None, downloads every registered dataset (about 6.7 GB).

    Returns:
        list of str: Names of the datasets that could not be downloaded
        (empty if all succeeded).
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
            all_datasets.append(DATASET_REGISTRY.get_dataset(name))
    total = len(all_datasets)
    cached = 0
    downloaded = 0
    failed = []

    for i, ds in enumerate(all_datasets, 1):
        # Look for a valid local copy first, so that "already cached" means
        # that the file is there and its hash is right
        candidates = [data_dir() / ds.get_filename(), DATA_PATH / ds.get_filename(), DATA_PATH / (hash_name(ds.name) + ".h5")]
        if any(_is_valid(c, ds.file_hash) for c in candidates):
            cached += 1
            print(f"[{i}/{total}] {ds.name} — already cached")
            continue

        print(f"[{i}/{total}] Downloading {ds.name}...")
        try:
            fetch_dataset(ds.name)
            downloaded += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(ds.name)

    print(f"\nDone: {downloaded} downloaded, {cached} already cached, {len(failed)} failed.")
    if failed:
        print(f"Failed datasets: {', '.join(failed)}")
    return failed


def create_dataset_file(file_name: str, data_info: dict, metadata: dict):
    """
    Create an HDF5 file containing multiple 1D profiles, each with a name, and include metadata.

    Args:
        file_name (str): The name of the HDF5 file to create.
        data_info (dict): A dictionary where keys are profile names and values are numpy arrays representing the profiles.
        metadata (dict): A dictionary containing metadata about the data source.

    The file is written to the package directory `DATA_PATH`, where developer
    scripts keep newly built datasets. The loader uses a file there if its
    hash matches the manifest.
    """
    # The package directory is not created on install, so create it here
    DATA_PATH.mkdir(parents=True, exist_ok=True)
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
