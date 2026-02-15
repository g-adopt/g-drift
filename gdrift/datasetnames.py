"""Dataset registry and manifest management for gdrift.

This module provides the centralized registry system for all datasets available
in gdrift. It reads from a JSON manifest file (`datasets.json`) that serves as
the single source of truth for dataset metadata, including names, types, file
hashes, citations, and S3 storage configuration.

The registry system enables:
- Type-safe dataset enumeration and validation
- SHA256 hash verification for data integrity
- Automatic derivation of available models (e.g., AVAILABLE_SEISMIC_MODELS)
- Filtering datasets by type or utility class
- Documentation generation via `print_datasets_markdown`

Architecture
------------
The manifest (`datasets.json`) contains:
1. S3/CDN configuration (bucket, prefix, endpoint, CDN URL)
2. Dataset entries array with metadata for each dataset

At import time, `_load_manifest()` reads the JSON and constructs a
`DatasetRegistry` populated with `Dataset` dataclass instances. This registry
is exposed as the module-level `DATASET_REGISTRY` singleton.

Key Classes
-----------
DatasetType : Enum of dataset categories (solidus, tomography, etc.)
UtilityClass : Enum of loading classes (SeismicModel, ThermodynamicModel, etc.)
Dataset : Dataclass representing a single dataset with metadata
DatasetRegistry : Collection of datasets with filtering and query methods

Key Functions
-------------
hash_name : Generate SHA256 hash of dataset name for obfuscated storage
get_manifest_config : Extract S3/CDN configuration from manifest
print_datasets_markdown : Generate markdown table of available datasets
_load_manifest : Internal loader for datasets.json

Module-Level Constants
----------------------
DATASET_REGISTRY : Singleton registry of all datasets
MANIFEST_PATH : Path to datasets.json file

Examples
--------
>>> import gdrift
>>> # List all datasets
>>> gdrift.print_datasets_markdown()
>>>
>>> # Filter by type
>>> tomography = gdrift.DATASET_REGISTRY.filter_by_type(
...     gdrift.datasetnames.DatasetType.TOMOGRAPHY_MODEL)
>>> print(f"Found {len(tomography)} tomography models")
>>>
>>> # Get specific dataset
>>> ds = gdrift.DATASET_REGISTRY.get("1d_prem")
>>> print(f"Type: {ds.dataset_type}, Citation: {ds.source}")

Notes
-----
- All 33 datasets in the manifest have real SHA256 hashes
- Dataset names must match the "name" field in datasets.json exactly
- Unknown dataset names raise ValueError in load_dataset()
- The manifest ships with the package (no external download needed)

See Also
--------
gdrift.io.load_dataset : Download and load datasets using the registry
"""

import json
import hashlib
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path


def hash_name(name: str) -> str:
    """Deterministic SHA-256 hash of a dataset name for obfuscated storage."""
    return hashlib.sha256(name.encode()).hexdigest()


class DatasetType(Enum):
    """Enumeration of dataset types in the gdrift registry.

    Categorizes datasets by their scientific purpose and data structure.
    Used for filtering datasets and deriving model lists (e.g.,
    AVAILABLE_SEISMIC_MODELS).

    Attributes
    ----------
    SOLIDUS_PROFILE : str
        1D solidus temperature profiles from experimental petrology
        (e.g., Andrault, Fiquet, Hirschmann).
    EARTH_MODEL : str
        1D reference Earth models with multiple radial profiles
        (e.g., PREM - density, velocity, gravity, pressure).
    GEODYNAMIC_PROFILE : str
        1D profiles from geodynamic adiabats (e.g., SLB_21 adiabatic
        temperature, density, velocity profiles).
    THERMODYNAMIC_MODEL : str
        2D thermodynamic lookup tables (depth × temperature) for mineral
        physics properties (e.g., SLB_21 pyroliteCFMAS).
    TOMOGRAPHY_MODEL : str
        3D seismic tomography models with velocity perturbations
        (e.g., S40RTS, GLAD-M25, SEMUCB-WM1).

    Examples
    --------
    >>> from gdrift.datasetnames import DATASET_REGISTRY, DatasetType
    >>> # Filter datasets by type
    >>> tomography = DATASET_REGISTRY.filter_by_type(DatasetType.TOMOGRAPHY_MODEL)
    >>> print(f"Found {len(tomography)} tomography models")
    """
    SOLIDUS_PROFILE = "1d Solidus Profile"
    EARTH_MODEL = "1d Reference Earth Models"
    GEODYNAMIC_PROFILE = "1d Geodynamic Profile"
    THERMODYNAMIC_MODEL = "Thermodynamic Model of Mantle Rocks"
    TOMOGRAPHY_MODEL = "Seismic Tomography Model"


class UtilityClass(Enum):
    """Enumeration of gdrift classes used to load datasets.

    Maps dataset types to their corresponding loader classes. Used in
    dataset metadata to indicate the recommended class for loading each
    dataset.

    Attributes
    ----------
    RADIAL_EARTH_MODEL : str
        "RadialEarthModelFromFile" - Generic loader for 1D radial profiles
        from HDF5 files (solidus, geodynamic adiabats).
    PREM : str
        "PreliminaryRefEarthModel" - Specialized loader for the PREM
        reference model (singleton, no arguments needed).
    THERMODYNAMIC : str
        "ThermodynamicModel" - Loader for 2D thermodynamic lookup tables
        (requires model and composition arguments).
    SEISMIC_MODEL : str
        "SeismicModel" - Loader for 3D seismic tomography models
        (requires model name argument).

    Examples
    --------
    >>> from gdrift.datasetnames import DATASET_REGISTRY, UtilityClass
    >>> # Filter datasets by utility class
    >>> seismic_ds = DATASET_REGISTRY.filter_by_utility(UtilityClass.SEISMIC_MODEL)
    >>> print(f"Found {len(seismic_ds)} datasets loadable via SeismicModel")

    Notes
    -----
    The utility field in Dataset objects is optional - some datasets (like
    the "test" dataset) have no associated loader class.
    """
    RADIAL_EARTH_MODEL = "RadialEarthModelFromFile"
    PREM = "PreliminaryRefEarthModel"
    THERMODYNAMIC = "ThermodynamicModel"
    SEISMIC_MODEL = "SeismicModel"


@dataclass
class Dataset:
    """
    Represents a scientific dataset with metadata and usage information.

    Attributes:
        name: Unique identifier for the dataset
        dataset_type: Type of the dataset (from DatasetType enum)
        source: Citation or reference information
        utility: Optional utility class for loading the dataset
        description: Optional description of the dataset
        doi: Optional DOI for the dataset
        year: Optional publication year
        file_hash: Optional SHA256 hash for file integrity verification
    """
    name: str
    dataset_type: DatasetType
    source: str
    utility: Optional[UtilityClass] = None
    description: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None
    file_hash: Optional[str] = None
    fields: Optional[List[str]] = None
    regional: Optional[bool] = None

    def __post_init__(self):
        """Validate dataset fields after dataclass initialization.

        Called automatically by the dataclass after __init__. Ensures
        that required fields (name, source) are populated and that
        values are within reasonable ranges.

        Raises
        ------
        ValueError
            If name or source is empty, source is "unknown", or year is
            outside the range [1900, 2030].

        Notes
        -----
        This validation runs at Dataset construction time, not when
        loading datasets from the manifest. Invalid manifest entries
        will cause import-time errors.
        """
        if not self.name:
            raise ValueError("Dataset name cannot be empty")
        if not self.source:
            raise ValueError("Dataset source cannot be empty")
        if self.source.lower() == "unknown":
            raise ValueError("Dataset source cannot be 'unknown' - please provide proper citation")
        if self.year is not None and (self.year < 1900 or self.year > 2030):
            raise ValueError("Year must be between 1900 and 2030")

    def __str__(self) -> str:
        utility_str = self.utility.value if self.utility else "No utility class specified"
        return f"{self.name} ({self.dataset_type.value}), How to use: {utility_str}, Cite: {self.source}"

    def __repr__(self) -> str:
        return f"Dataset(name='{self.name}', type={self.dataset_type.name}, utility={self.utility.name if self.utility else None})"

    def is_compatible_with_utility(self, utility_class: UtilityClass) -> bool:
        """Check if this dataset is compatible with a given utility class."""
        return self.utility == utility_class

    def get_citation_info(self) -> Dict[str, Any]:
        """Get structured citation information."""
        return {
            "source": self.source,
            "doi": self.doi,
            "year": self.year,
            "dataset_name": self.name
        }

    def get_filename(self) -> str:
        """Get the expected filename for this dataset (obfuscated via hash)."""
        return f"{hash_name(self.name)}.h5"

    def has_hash(self) -> bool:
        """Check if this dataset has a hash for integrity verification."""
        return self.file_hash is not None


class DatasetRegistry:
    """Registry for managing and querying datasets.

    Central repository of all registered datasets in gdrift. Provides methods
    for filtering, searching, and validating dataset metadata. Constructed
    from the datasets.json manifest at module import time.

    Parameters
    ----------
    datasets : list of Dataset
        List of Dataset objects to register. Names must be unique.

    Attributes
    ----------
    _datasets : dict
        Internal dictionary mapping dataset names to Dataset objects.

    Examples
    --------
    >>> from gdrift.datasetnames import DATASET_REGISTRY, DatasetType
    >>> # Get all dataset names
    >>> print(DATASET_REGISTRY.get_dataset_names())
    >>> # Filter by type
    >>> solidus_profiles = DATASET_REGISTRY.filter_by_type(
    ...     DatasetType.SOLIDUS_PROFILE)
    >>> # Search by pattern
    >>> slb_datasets = DATASET_REGISTRY.search_by_name("slb")
    >>> # Check existence
    >>> if "1d_prem" in DATASET_REGISTRY:
    ...     prem = DATASET_REGISTRY.get_dataset("1d_prem")

    Notes
    -----
    The module-level `DATASET_REGISTRY` singleton is initialized automatically
    by reading `gdrift/datasets.json`. Users should not need to create
    DatasetRegistry instances manually.
    """

    def __init__(self, datasets: List[Dataset]):
        """Initialize the registry with a list of datasets.

        Parameters
        ----------
        datasets : list of Dataset
            List of Dataset objects to register.

        Raises
        ------
        ValueError
            If duplicate dataset names are found.
        """
        self._datasets = {dataset.name: dataset for dataset in datasets}
        self._validate_unique_names(datasets)

    def _validate_unique_names(self, datasets: List[Dataset]):
        """Ensure all dataset names are unique."""
        names = [dataset.name for dataset in datasets]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate dataset names found: {duplicates}")

    def get_dataset(self, name: str) -> Optional[Dataset]:
        """Get a dataset by name."""
        return self._datasets.get(name)

    def list_datasets(self) -> List[Dataset]:
        """Get all datasets."""
        return list(self._datasets.values())

    def filter_by_type(self, dataset_type: DatasetType) -> List[Dataset]:
        """Filter datasets by type."""
        return [dataset for dataset in self._datasets.values()
                if dataset.dataset_type == dataset_type]

    def filter_by_utility(self, utility: UtilityClass) -> List[Dataset]:
        """Filter datasets by utility class."""
        return [dataset for dataset in self._datasets.values()
                if dataset.utility == utility]

    def search_by_name(self, pattern: str) -> List[Dataset]:
        """Search datasets by name pattern (case-insensitive)."""
        pattern_lower = pattern.lower()
        return [dataset for dataset in self._datasets.values()
                if pattern_lower in dataset.name.lower()]

    def get_dataset_names(self) -> List[str]:
        """Get all dataset names."""
        return list(self._datasets.keys())

    def get_datasets_with_hashes(self) -> List[Dataset]:
        """Get all datasets that have hash information for integrity verification."""
        return [dataset for dataset in self._datasets.values() if dataset.has_hash()]

    def filter_by_field(self, field: str) -> List[Dataset]:
        """Filter datasets containing a specific field (e.g., 'dvs')."""
        return [ds for ds in self._datasets.values()
                if ds.fields and field in ds.fields]

    def __contains__(self, name: str) -> bool:
        """Check if a dataset exists in the registry."""
        return name in self._datasets

    def __len__(self) -> int:
        """Get the number of datasets."""
        return len(self._datasets)


_MANIFEST_PATH = Path(__file__).parent / "datasets.json"
_manifest_cache = None


def _load_manifest():
    """Load the datasets.json manifest and return the parsed dict."""
    global _manifest_cache
    if _manifest_cache is None:
        with open(_MANIFEST_PATH, "r") as f:
            _manifest_cache = json.load(f)
    return _manifest_cache


def get_manifest_config() -> Dict[str, str]:
    """Return S3 configuration from the manifest for use by io.py."""
    manifest = _load_manifest()
    return {
        "endpoint_url": manifest["s3"]["endpoint_url"],
        "bucket": manifest["s3"]["bucket"],
        "prefix": manifest["s3"]["prefix"],
        "cdn_url": manifest.get("cdn_url", ""),
    }


def _build_registry_from_manifest() -> List[Dataset]:
    """Build Dataset objects from the JSON manifest."""
    manifest = _load_manifest()
    datasets = []
    for entry in manifest["datasets"]:
        ds = Dataset(
            name=entry["name"],
            dataset_type=DatasetType[entry["type"]],
            source=entry["source"],
            utility=UtilityClass[entry["utility"]] if entry.get("utility") else None,
            description=entry.get("description"),
            doi=entry.get("doi"),
            year=entry.get("year"),
            file_hash=f"sha256:{entry['sha256']}" if entry.get("sha256") else None,
            fields=entry.get("fields"),
            regional=entry.get("regional"),
        )
        datasets.append(ds)
    return datasets


# Build the registry and list from the manifest at module level
_datasets_list = _build_registry_from_manifest()
DATASET_REGISTRY = DatasetRegistry(_datasets_list)
AVAILABLE_DATASETS = _datasets_list


def print_datasets_markdown():
    """Print a markdown table of all available datasets."""
    datasets = AVAILABLE_DATASETS
    print("| Name | Type | How to Use | Year | Has Hash | Source |")
    print("|------|------|------------|------|----------|--------|")
    for dataset in datasets:
        utility_str = dataset.utility.value if dataset.utility else "No utility class specified"
        year_str = str(dataset.year) if dataset.year else "N/A"
        hash_str = "Y" if dataset.has_hash() else "N"
        print(f"| {dataset.name} | {dataset.dataset_type.value} | {utility_str} | {year_str} | {hash_str} | {dataset.source} |")


def get_datasets_by_type(dataset_type: DatasetType) -> List[Dataset]:
    """Get all datasets of a specific type."""
    return DATASET_REGISTRY.filter_by_type(dataset_type)


def get_dataset_by_name(name: str) -> Optional[Dataset]:
    """Get a dataset by name."""
    return DATASET_REGISTRY.get_dataset(name)


def search_datasets(pattern: str) -> List[Dataset]:
    """Search for datasets by name pattern."""
    return DATASET_REGISTRY.search_by_name(pattern)


def get_dataset_hash(dataset_name: str) -> Optional[str]:
    """Get the hash for a dataset by name."""
    dataset = get_dataset_by_name(dataset_name)
    return dataset.file_hash if dataset else None
