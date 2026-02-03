import json
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path


class DatasetType(Enum):
    """Enumeration of available dataset types."""
    SOLIDUS_PROFILE = "1d Solidus Profile"
    EARTH_MODEL = "1d Reference Earth Models"
    GEODYNAMIC_PROFILE = "1d Geodynamic Profile"
    THERMODYNAMIC_MODEL = "Thermodynamic Model of Mantle Rocks"
    TOMOGRAPHY_MODEL = "Seismic Tomography Model"


class UtilityClass(Enum):
    """Enumeration of utility classes for loading datasets."""
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

    def __post_init__(self):
        """Validate the dataset after initialization."""
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
        """Get the expected filename for this dataset."""
        return f"{self.name}.h5"

    def has_hash(self) -> bool:
        """Check if this dataset has a hash for integrity verification."""
        return self.file_hash is not None


class DatasetRegistry:
    """Registry for managing and querying datasets."""

    def __init__(self, datasets: List[Dataset]):
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
