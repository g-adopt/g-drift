# Different types
_TYPE_SOLIDUS_PROFILE = "1d Solidus Profile"
_TYPE_EARTH_MODEL = "1d Reference Earth Models"
_TYPE_THERMODYNAMIC_MODEL = "Thermodynamic Model of Mantle Rocks"
_TYPE_TOMOGRAPHY_MODEL = "Seismic Tomography Model"

# What utility should be used for what
_UTIL_SOLIDUS = "SolidusProfileFromFile"
_UTIL_PREM = "PreliminaryRefEarthModel"
_UTIL_THERMODYNAMIC = "ThermodynamicModel"


class Dataset:
    def __init__(self, name, dataset_type, source, utility):
        self.name = name
        self.dataset_type = dataset_type
        self.source = source
        self.utility = utility

    def __str__(self):
        return f"{self.name} ({self.dataset_type}), How to use: {self.utility}, Cite: {self.source}"


AVAILABLE_DATASETS = [
    Dataset(
        name="1d_prem",
        dataset_type=_TYPE_EARTH_MODEL,
        source="Dziewonski, Adam M., and Don L. Anderson. 'Preliminary reference Earth model.' Physics of the earth and planetary interiors 25.4 (1981): 297-356.",
        utility=_UTIL_PREM),
    Dataset(
        name="1d_solidus_Andrault_et_al_2011_EPSL",
        dataset_type=_TYPE_SOLIDUS_PROFILE,
        source="Andrault, Denis, et al. 'Solidus and liquidus profiles of chondritic mantle: Implication for melting of the Earth across its history.' Earth and planetary science letters 304.1-2 (2011): 251-259.",
        utility=_UTIL_SOLIDUS),
    Dataset(
        name="1d_solidus_Fiquet_et_al_2010_SCIENCE",
        dataset_type=_TYPE_SOLIDUS_PROFILE,
        source='Fiquet, G., et al. "Melting of peridotite to 140 gigapascals." Science 329.5998 (2010): 1516-1518.',
        utility=_UTIL_SOLIDUS),
    Dataset(
        name="1d_solidus_Ghelichkhan_et_al_2021_GJI",
        dataset_type=_TYPE_SOLIDUS_PROFILE,
        source="Ghelichkhan et al., 2021, GJI",
        utility=_UTIL_SOLIDUS),
    Dataset(
        name="1d_solidus_Nomura_et_al_2014_SCIENCE",
        dataset_type=_TYPE_SOLIDUS_PROFILE,
        source='Nomura, Ryuichi, et al. "Low core-mantle boundary temperature inferred from the solidus of pyrolite." Science 343.6170 (2014): 522-525.',
        utility=_UTIL_SOLIDUS),
    Dataset(
        name="1d_solidus_Zerr_et_al_1988_SCIENCE",
        dataset_type=_TYPE_SOLIDUS_PROFILE,
        source='Zerr, A., and R. Boehler. "Constraints on the melting temperature of the lower mantle from high-pressure experiments on MgO and magnesioüstite." Nature 371.6497 (1994): 506-508.',
        utility=_UTIL_SOLIDUS),
    Dataset(
        name="REVEAL",
        dataset_type=_TYPE_TOMOGRAPHY_MODEL,
        source='Thrastarson, Solvi, et al. "REVEAL: A global full‐waveform inversion model." Bulletin of the Seismological Society of America 114.3 (2024): 1392-1406.',
        utility=None),
    Dataset(
        name="SLB_16_basalt",
        dataset_type=_TYPE_THERMODYNAMIC_MODEL,
        source="Unknown",
        utility=None),
    Dataset(
        name="SLB_16_pyrolite",
        dataset_type=_TYPE_THERMODYNAMIC_MODEL,
        source="Unknown",
        utility=None)
]


def print_datasets_markdown():
    datasets = AVAILABLE_DATASETS
    print("| Name | Type | How to Use | Source |")
    print("|------|------|------------|--------|")
    for dataset in datasets:
        print(
            f"| {dataset.name} | {dataset.dataset_type} | {dataset.utility} | {dataset.source} |")
