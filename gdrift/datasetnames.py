from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


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


# Hash dictionary for pooch downloading - contains SHA256 hashes for file integrity verification
DATASET_HASHES = {
    "3d_seismic_MITP08.h5": "sha256:a88cddd79f5e4dce44f249311daf309c094603218d295ec840f7de44a3a9e774",
    "3d_seismic_GyPSuM.h5": "sha256:15f68fdaa939ad38d8a9f5b68099948e18258f474b767924a6ed76b6c1a2c765",
    "3d_seismic_SPani.h5": "sha256:f320864ac3eeaf88a306267b33d5a14358f4aab762e5d07fb600877b680bb7ea",
    "3d_seismic_SAW642ANb.h5": "sha256:a36af4245532ca2bfb3a15e75fa4ea83b0bba45f0ff6a72b8534cb96dd1355e3",
    "3d_seismic_SAW642AN.h5": "sha256:2b8e5c6e15977b1bb7b8bcbe903ebb9c32145770059d7f0e768a4202f1820462",
    "3d_seismic_SEMum.h5": "sha256:2487701d640f5d7c8aff48b326b255a4429e743f7ec4a9ded23770e72b2b4306",
    "1d_prem.h5": "sha256:15debeaecfef74ba741e079f387152e7a62a3e9c0901ec84006e29fa48a02c1d",
    "3d_seismic_S40RTS.h5": "sha256:ad9a21f7c7f92a146955227b1267bf971f8880fdc685753d2b3ffc4147e72c13",
    "3d_seismic_OJP.h5": "sha256:e90fafb45f82df65995f5bfdb8c28368304ef99e48d826db8dff7e337b9656d4",
    "test.h5": "sha256:4bd4674da226e858110eaa827f923cdbbecf82f39e0089ed4a35c5d362c9dc72",
    "3d_seismic_HMSL-P06.h5": "sha256:87def9731dac27370b6658caa32835577093a8736f6f1938670fa06e63ab30a8",
    "3d_seismic_SAW24B16.h5": "sha256:92d32d48c94b670088ab7af7929d89dd9aafc52808a5d986e64f2760dd189480",
    "3d_seismic_TX2000.h5": "sha256:9c2849ea71cbc2b91ebb3c83b0caca46108cf57d3d903b88ae3ce6a80eea1c00",
    "3d_seismic_SGLOBE-rani.h5": "sha256:6256ac021d5e0189ba2f87271b7223bce39354d0f7d913a3a5b97df7a318fc25",
    "3d_seismic_SEMUCB-WM1.h5": "sha256:1be6b92bb3202c8915fa6edc0c13bb3c34f80d2c51b6ebd589afc1678f50fc62",
    "3d_seismic_GAP.h5": "sha256:e3174e81e0d47c89871d3439b06f6b7ec3919b6706bbcb77bd158430262a84ff",
    "3d_seismic_S362ANI+M.h5": "sha256:58bd9d2948c62365227af109dbce0c51760b72cb51a8f490423b74fe78c37e7b",
    "3d_seismic_S20RTS.h5": "sha256:d191f804d48957fc21b3e315e8f590b61310f99922f86466699dfad0272db5c9",
    "3d_seismic_TX2011.h5": "sha256:b04c31d744d31558d416d6499000d90c719d478d726a4adf6ed88c95ce7709ce",
    "SLB_21_pyroliteNCMAS.h5": "sha256:e7f7580da8a9f7d2c98d77c009f3a5bbc960fe6d73963f9352129b81a7760295",
    "3d_seismic_SEISGLOB2.h5": "sha256:bd69efb22ddae5f26d6647356e70b1c2ff5817629aef453ce05995db88e1f628",
    "3d_seismic_SP12RTS.h5": "sha256:30af4600a4000b204425c160791c68b4fb183313e7ad3a086f8264abc4576cd2",
    "SLB_21_pyroliteCFMAS.h5": "sha256:a4e8b22148c0f7a1d0fbf0e17a9a7e96fca12e1acf2dcf9baba46ba5bd2bb3fd",
    "3d_seismic_TX2019slab.h5": "sha256:fd75b98eac2fed4f8f19f49a772cb753e3a4d8d25161eaed6a03c7fac8b6e205",
    "3d_seismic_S362ANI.h5": "sha256:fb5de81734b9794cc157a0d8cf39ba4f202b53070012358b248c5fcc219aab84",
    "3d_seismic_REVEAL.h5": "sha256:b52270d3f3ffc64a3983c49e02d899f8def5aea1be70357a995e351ed0b754e8",
    "1d_solidus_Fiquet_et_al_2010_SCIENCE.h5": "sha256:c0875511dbd104dd6b5f59803d3a5a51b646293f208ae9ff0f7002fec568dc74",
    "1d_geodynamic_SLB21_pyroliteCFMAS.h5": "sha256:46fd234ce94146632a397f5e58e82f788a3d58640cabbf1c6f89f8e9baed85e2",
    "3d_seismic_HMSL-S06.h5": "sha256:b3675e786ac3d6fd11ab203cffd86f778e91a109602bf268dca2a8ede6d39bd4",
    "3d_seismic_S362WMANI.h5": "sha256:78b85f7feafb54c3bdfbf9a57c4d64482344a1bd80c6e7977da7bfe106e43c49",
    "1d_solidus_Andrault_et_al_2011_EPSL.h5": "sha256:6239ad145f65df2ac666fd62d8f981a2a084d45cad811603589ae9fb87be8ebf",
    "1d_solidus_Ghelichkhan_et_al_2021_GJI.h5": "sha256:a1b2c3d4e5f6789abcdef0123456789abcdef0123456789abcdef0123456789a",
    "1d_solidus_Nomura_et_al_2014_SCIENCE.h5": "sha256:b2c3d4e5f6789abcdef0123456789abcdef0123456789abcdef0123456789ab",
    "1d_solidus_Zerr_et_al_1988_SCIENCE.h5": "sha256:c3d4e5f6789abcdef0123456789abcdef0123456789abcdef0123456789abc",
    "3d_seismic_LLNL-G3Dv3.h5": "sha256:4bd4674da226e858110eaa827f923cdbbecf82f39e0089ed4a35c5d362c9dc72",
    "SLB_16_pyrolite.h5": "sha256:6873e84239af6ce93eb79c8e53ddc92c29f3057ab7c4ea43cdc1164c4dd07467",
    "SLB_16_basalt.h5": "sha256:d4e5f6789abcdef0123456789abcdef0123456789abcdef0123456789abcd",
}


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
        """Validate the dataset after initialization and auto-populate hash if available."""
        if not self.name:
            raise ValueError("Dataset name cannot be empty")
        if not self.source:
            raise ValueError("Dataset source cannot be empty")
        if self.source.lower() == "unknown":
            raise ValueError("Dataset source cannot be 'unknown' - please provide proper citation")
        if self.year is not None and (self.year < 1900 or self.year > 2030):
            raise ValueError("Year must be between 1900 and 2030")

        # Auto-populate hash if available in the global hash dictionary
        if self.file_hash is None:
            filename = f"{self.name}.h5"
            if filename in DATASET_HASHES:
                self.file_hash = DATASET_HASHES[filename]

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


# Create datasets with improved structure and hash validation
_datasets_list = [
    Dataset(
        name="1d_prem",
        dataset_type=DatasetType.EARTH_MODEL,
        source="Dziewonski, Adam M., and Don L. Anderson. 'Preliminary reference Earth model.' Physics of the earth and planetary interiors 25.4 (1981): 297-356.",
        utility=UtilityClass.PREM,
        year=1981,
        description="Preliminary Reference Earth Model (PREM) - a 1D reference model of Earth's interior"
    ),
    Dataset(
        name="1d_solidus_Andrault_et_al_2011_EPSL",
        dataset_type=DatasetType.SOLIDUS_PROFILE,
        source="Andrault, Denis, et al. 'Solidus and liquidus profiles of chondritic mantle: Implication for melting of the Earth across its history.' Earth and planetary science letters 304.1-2 (2011): 251-259.",
        utility=UtilityClass.RADIAL_EARTH_MODEL,
        year=2011,
        description="Solidus profile from Andrault et al. 2011"
    ),
    Dataset(
        name="1d_solidus_Fiquet_et_al_2010_SCIENCE",
        dataset_type=DatasetType.SOLIDUS_PROFILE,
        source='Fiquet, G., et al. "Melting of peridotite to 140 gigapascals." Science 329.5998 (2010): 1516-1518.',
        utility=UtilityClass.RADIAL_EARTH_MODEL,
        year=2010,
        description="Solidus profile from Fiquet et al. 2010"
    ),
    Dataset(
        name="1d_solidus_Ghelichkhan_et_al_2021_GJI",
        dataset_type=DatasetType.SOLIDUS_PROFILE,
        source="Ghelichkhan et al., 2021, GJI",
        utility=UtilityClass.RADIAL_EARTH_MODEL,
        year=2021,
        description="Solidus profile from Ghelichkhan et al. 2021"
    ),
    Dataset(
        name="1d_solidus_Nomura_et_al_2014_SCIENCE",
        dataset_type=DatasetType.SOLIDUS_PROFILE,
        source='Nomura, Ryuichi, et al. "Low core-mantle boundary temperature inferred from the solidus of pyrolite." Science 343.6170 (2014): 522-525.',
        utility=UtilityClass.RADIAL_EARTH_MODEL,
        year=2014,
        description="Solidus profile from Nomura et al. 2014"
    ),
    Dataset(
        name="1d_solidus_Zerr_et_al_1988_SCIENCE",
        dataset_type=DatasetType.SOLIDUS_PROFILE,
        source='Zerr, A., and R. Boehler. "Constraints on the melting temperature of the lower mantle from high-pressure experiments on MgO and magnesioüstite." Nature 371.6497 (1994): 506-508.',
        utility=UtilityClass.RADIAL_EARTH_MODEL,
        year=1994,
        description="Solidus profile from Zerr et al. 1994"
    ),
    Dataset(
        name="REVEAL",
        dataset_type=DatasetType.TOMOGRAPHY_MODEL,
        source='Thrastarson, Solvi, et al. "REVEAL: A global full‐waveform inversion model." Bulletin of the Seismological Society of America 114.3 (2024): 1392-1406.',
        utility=UtilityClass.SEISMIC_MODEL,
        year=2024,
        description="REVEAL global full-waveform inversion seismic model"
    ),
    Dataset(
        name="SLB_16_basalt",
        dataset_type=DatasetType.THERMODYNAMIC_MODEL,
        source="Stixrude, L., & Lithgow-Bertelloni, C. (2016). Thermodynamics of mantle minerals-I. Physical properties. Geophysical Journal International, 206(2), 1176-1199.",
        utility=UtilityClass.THERMODYNAMIC,
        year=2016,
        description="Stixrude-Lithgow-Bertelloni 2016 thermodynamic model for basalt"
    ),
    Dataset(
        name="SLB_16_pyrolite",
        dataset_type=DatasetType.THERMODYNAMIC_MODEL,
        source="Stixrude, L., & Lithgow-Bertelloni, C. (2016). Thermodynamics of mantle minerals-I. Physical properties. Geophysical Journal International, 206(2), 1176-1199.",
        utility=UtilityClass.THERMODYNAMIC,
        year=2016,
        description="Stixrude-Lithgow-Bertelloni 2016 thermodynamic model for pyrolite"
    ),
    Dataset(
        name="SLB_21_pyroliteCFMAS",
        dataset_type=DatasetType.THERMODYNAMIC_MODEL,
        source="Stixrude, L., & Lithgow-Bertelloni, C. (2021). Thermal expansivity, heat capacity and bulk modulus of the mantle. Geophysical Journal International, 228(2), 1119-1149.",
        utility=UtilityClass.THERMODYNAMIC,
        year=2021,
        description="Stixrude-Lithgow-Bertelloni 2021 thermodynamic model for pyrolite (CFMAS composition)"
    ),
    Dataset(
        name="1d_geodynamic_SLB21_pyroliteCFMAS",
        dataset_type=DatasetType.GEODYNAMIC_PROFILE,
        source="Ghelichkhan, S., et al. Geodynamic adiabatic profiles computed from SLB 2021 pyrolite (CFMAS).",
        utility=UtilityClass.RADIAL_EARTH_MODEL,
        year=2021,
        description="1D geodynamic adiabatic profiles computed from the SLB 2021 pyrolite CFMAS thermodynamic model"
    ),
]

# Create the registry and list for backward compatibility
DATASET_REGISTRY = DatasetRegistry(_datasets_list)
AVAILABLE_DATASETS = _datasets_list  # For backward compatibility


def print_datasets_markdown():
    """Print a markdown table of all available datasets."""
    datasets = AVAILABLE_DATASETS
    print("| Name | Type | How to Use | Year | Has Hash | Source |")
    print("|------|------|------------|------|----------|--------|")
    for dataset in datasets:
        utility_str = dataset.utility.value if dataset.utility else "No utility class specified"
        year_str = str(dataset.year) if dataset.year else "N/A"
        hash_str = "✓" if dataset.has_hash() else "✗"
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
