"""Three-dimensional seismic tomography models with spatial interpolation.

This module provides access to the global and regional seismic tomography
models of the dataset registry, with shear-wave (Vs) and compressional-wave
(Vp) speeds or their perturbations throughout the mantle. Models are loaded from HDF5 datasets stored on remote
S3-compatible storage and interpolated using KD-tree nearest-neighbor search
with configurable kernels.

Seismic tomography models represent the 3D distribution of velocity anomalies
(dVs, dVp) relative to a 1D reference model. These perturbations are interpreted
as thermal and compositional heterogeneity in mantle convection studies.

Available Models
----------------
The module dynamically constructs `AVAILABLE_SEISMIC_MODELS` from the dataset
registry. The models include:
- S40RTS (Ritsema et al., 2011) - widely used Vs model
- GLAD-M35 (Cui et al., 2024) - joint Vs/Vp adjoint model with uncertainties
- SEMUCB-WM1 (French & Romanowicz, 2014) - full mantle Vs
- SAW642AN (Panning & Romanowicz, 2006) - radially anisotropic
- Multiple regional and global P-wave models

Key Classes
-----------
SeismicModel : Load and query 3D seismic tomography models

Key Functions
-------------
AVAILABLE_SEISMIC_MODELS : List of available model names (derived from registry)

Examples
--------
>>> import gdrift
>>> # List available models (names are case-sensitive)
>>> print(gdrift.AVAILABLE_SEISMIC_MODELS[:5])
['GAP', 'GyPSuM', 'HMSL-P06', 'HMSL-S06', 'LLNL-G3Dv3']
>>>
>>> # Load S40RTS model
>>> s40rts = gdrift.SeismicModel("S40RTS", nearest_neighbours=8)
>>> # Query Vs (m/s) at a location given as lat, lon (degrees) and depth (m)
>>> xyz = gdrift.geodetic_to_cartesian(45.0, 10.0, 1000e3)
>>> vs = s40rts.at("vs", xyz)

Notes
-----
- Model names should be provided WITHOUT the "3d_seismic_" prefix
  (e.g., use "S40RTS", not "3d_seismic_S40RTS")
- Coordinates: lat/lon in degrees, depth in meters from surface
- Interpolation is limited to `maximum_distance` (200 km) from data points
- Below `minimum_distance` (1 m), returns exact values without interpolation

See Also
--------
gdrift.EarthModel3D : Base class with interpolation infrastructure
gdrift.load_dataset : Download and cache HDF5 datasets
"""

from .earthmodel3d import EarthModel3D
from .datasetnames import DATASET_REGISTRY, DatasetType

# Prefix of the registry names of seismic tomography datasets. Users give the
# name without it.
SEISMIC_MODEL_PREFIX = "3d_seismic_"

AVAILABLE_SEISMIC_MODELS = [
    ds.name.removeprefix(SEISMIC_MODEL_PREFIX)
    for ds in DATASET_REGISTRY.filter_by_type(DatasetType.TOMOGRAPHY_MODEL)
]


class SeismicModel(EarthModel3D):
    # Hard coding minimum distance, below which we do not interpolate
    minimum_distance = 1e-3
    # Hard coding maximum distance beyond which we don't have access to data
    maximum_distance = 200e3
    # Default interpolation kernel — Wendland C2 compact support avoids
    # contamination from distant neighbors in coarse/irregular grids.
    default_kernel = "wendland"

    def __init__(self, model_name, nearest_neighbours: int = 8, default_max_distance: float = 200e3, labels=[]):
        """SeismicModel is a class for handling 3D seismic models.

        This class inherits from EarthModel3D and is used to load and manage seismic models.
        It allows for the initialization of a seismic model with a specified name, number of nearest neighbours,
        and a default maximum distance. The model data is loaded from a dataset file, and quantities and coordinates
        are set accordingly.

        Parameters:
        -----------
        model_name : str
            Name of the seismic model to load
        nearest_neighbours : int, optional
            Number of nearest neighbours to consider for interpolation. Defaults to 8.
        default_max_distance : float, optional
            The default maximum distance for the model in meters. Defaults to 200e3.
        labels : list, optional
            Specific labels to load from the dataset. If empty, loads all available fields.

        Attributes:
            model_name (str): The name of the seismic model.
            nearest_neighbours (int): The number of nearest neighbours to consider.
            default_max_distance (float): The default maximum distance for the model in meters.
        """
        if model_name not in AVAILABLE_SEISMIC_MODELS:
            raise ValueError(f"Model '{model_name}' not found in available models. Choose from: {', '.join(AVAILABLE_SEISMIC_MODELS)}")

        self.model_name = model_name

        super().__init__(nearest_neighbours=nearest_neighbours, default_max_distance=default_max_distance)

        # Load the model data
        self._load_fields(labels=labels)

    @classmethod
    def from_dataset(cls, model_name, nearest_neighbours=8, default_max_distance=200e3, labels=None):
        """Load a seismic model by name; the same as ``SeismicModel(model_name, ...)``.

        `EarthModel3D.from_dataset` loads "3d_earthmodel_" datasets. This
        override keeps the method meaningful on the subclass, where it loads
        "3d_seismic_" datasets through the constructor.

        Parameters
        ----------
        model_name : str
            Name without the "3d_seismic_" prefix, one of
            `AVAILABLE_SEISMIC_MODELS`.
        nearest_neighbours : int, optional
            Number of neighbours for the interpolation. Default is 8.
        default_max_distance : float, optional
            Distance in metres beyond which a query returns NaN. Default is 200e3.
        labels : list of str, optional
            Load only these quantities. Default is all quantities.

        Returns
        -------
        SeismicModel
            The loaded model.
        """
        return cls(model_name, nearest_neighbours=nearest_neighbours,
                   default_max_distance=default_max_distance, labels=labels or [])

    def _load_fields(self, labels=[]):
        """
        Load fields from the seismic model dataset.

        Parameters:
        -----------
        labels : list
            Specific labels to load. If empty, loads all available fields.
        """
        # The file layout is the same as for every gridded 3D dataset, so the
        # reading is done by EarthModel3D; only the registry prefix differs.
        self._load_dataset_fields(f"{SEISMIC_MODEL_PREFIX}{self.model_name}", labels=labels)
