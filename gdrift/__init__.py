"""gdrift: Geodynamics Data Reformatting and Integration Facilitation Toolkit.

gdrift provides a unified interface for loading, processing, and converting
geodynamic and seismic data used in large-scale mantle convection studies.
The package bridges the gap between diverse data formats and computational
workflows, enabling seamless integration of 1D reference models, 3D seismic
tomography, thermodynamic lookup tables, and anelastic corrections.

Architecture
------------
The package is organized into several key modules:

- **Data Management**: Automatic download from S3-compatible storage with
  SHA256 verification and local caching (io.py, datasetnames.py)
- **1D Profiles**: Radial Earth models and solidus profiles with spline
  interpolation (profile.py)
- **3D Models**: KD-tree based spatial interpolation with multiple kernel
  functions for seismic tomography (earthmodel3d.py, seismic.py)
- **Thermodynamics**: 2D lookup tables for depth-temperature property queries
  (mineralogy.py)
- **Anelasticity**: Frequency-dependent velocity corrections for converting
  between elastic and anelastic models (anelasticity.py)
- **Utilities**: Coordinate transforms, gravity computation, and helper
  functions (utility.py)

Key Classes
-----------
ThermodynamicModel : 2D lookup tables for mineral physics properties
SeismicModel : 3D seismic tomography model with spatial interpolation
EarthModel3D : Generic 3D data container with KD-tree interpolation
PreliminaryRefEarthModel : PREM radial reference model
RadialEarthModelFromFile : Load custom 1D profiles from HDF5
CammaranoAnelasticityModel : Anelastic corrections (B, g parameterization)
GoesAnelasticityModel : Anelastic corrections (Q0, xi parameterization)

Key Functions
-------------
load_dataset : Download and load datasets from remote storage
apply_anelastic_correction : Apply frequency-dependent velocity corrections
regularise_thermodynamic_table : Smooth phase transitions in lookup tables
compute_gravity : Radial gravity profile from density
compute_pressure : Depth-integrated hydrostatic pressure
geodetic_to_cartesian : Spherical to Cartesian coordinate transform

Examples
--------
>>> import gdrift
>>> # Load PREM reference model
>>> prem = gdrift.PreliminaryRefEarthModel()
>>> rho_profile = prem.get_profile("density")
>>> density_at_670km = rho_profile.at_depth(670e3)
>>>
>>> # Load thermodynamic lookup table
>>> tm = gdrift.ThermodynamicModel("SLB_21_pyroliteCFMAS")
>>> vs = tm.temperature_to_vs(temperature=1600, depth=500e3)
>>>
>>> # Load seismic tomography
>>> s40rts = gdrift.SeismicModel("3d_seismic_s40rts")
>>> dvs = s40rts.at(lat=45, lon=0, depth=1000e3, quantity="dvs")

Notes
-----
Datasets are stored on Digital Ocean Spaces (S3-compatible) and downloaded
on first use. All downloads are verified against SHA256 hashes stored in
the manifest (gdrift/datasets.json). Use `gdrift.print_datasets_markdown()`
to see available datasets.

See Also
--------
print_datasets_markdown : List all registered datasets
DATASET_REGISTRY : Registry of available datasets with metadata
AVAILABLE_SEISMIC_MODELS : List of 25 seismic tomography models
"""

from .adiabat import compute_adiabat, prem_gravity_profile
from .anelasticity import CammaranoAnelasticityModel, GoesAnelasticityModel, apply_anelastic_correction
from .constants import R_earth, R_cmb
from .datasetnames import print_datasets_markdown, DATASET_REGISTRY
from .earthmodel3d import EarthModel3D
from .io import load_dataset, create_dataset_file, download_all_datasets
from .mineralogy import ThermodynamicModel, compute_pwave_speed, compute_swave_speed, regularise_thermodynamic_table
from .profile import PreliminaryRefEarthModel, RadialEarthModelFromFile, HirschmannSolidus, SplineProfile
from .utility import compute_gravity, compute_mass, compute_pressure, geodetic_to_cartesian, cartesian_to_geodetic, dimensionalise_coords, nondimensionalise_coords, fibonacci_sphere, great_circle_path, great_circle_cross_section
from .seismic import SeismicModel, AVAILABLE_SEISMIC_MODELS
from .gplates import CoastlineVTKFile

__all__ = [
    "compute_adiabat",
    "prem_gravity_profile",
    "CammaranoAnelasticityModel",
    "GoesAnelasticityModel",
    "apply_anelastic_correction",
    "R_earth",
    "R_cmb",
    "print_datasets_markdown",
    "EarthModel3D",
    "load_dataset",
    "create_dataset_file",
    "download_all_datasets",
    "ThermodynamicModel",
    "compute_pwave_speed",
    "compute_swave_speed",
    "regularise_thermodynamic_table",
    "PreliminaryRefEarthModel",
    "RadialEarthModelFromFile",
    "HirschmannSolidus",
    "SplineProfile",
    "compute_gravity",
    "compute_mass",
    "compute_pressure",
    "geodetic_to_cartesian",
    "cartesian_to_geodetic",
    "dimensionalise_coords",
    "nondimensionalise_coords",
    "fibonacci_sphere",
    "great_circle_path",
    "great_circle_cross_section",
    "SeismicModel",
    "AVAILABLE_SEISMIC_MODELS",
    "DATASET_REGISTRY",
    "CoastlineVTKFile",
]
