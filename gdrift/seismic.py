"""Three-dimensional seismic tomography models with spatial interpolation.

This module provides access to 25 global seismic tomography models covering
both shear wave (Vs) and compressional wave (Vp) velocity perturbations
throughout the mantle. Models are loaded from HDF5 datasets stored on remote
S3-compatible storage and interpolated using KD-tree nearest-neighbor search
with configurable kernels.

Seismic tomography models represent the 3D distribution of velocity anomalies
(dVs, dVp) relative to a 1D reference model. These perturbations are interpreted
as thermal and compositional heterogeneity in mantle convection studies.

Available Models
----------------
The module dynamically constructs `AVAILABLE_SEISMIC_MODELS` from the dataset
registry. As of the current manifest, 25 models are available including:
- S40RTS (Ritsema et al., 2011) - widely used Vs model
- GLAD-M25 (Lei et al., 2020) - joint Vs/Vp model
- SEMUCB-WM1 (French & Romanowicz, 2014) - full mantle Vs
- SAW642AN (Panning et al., 2010) - azimuthally anisotropic
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
>>> # List available models
>>> print(gdrift.AVAILABLE_SEISMIC_MODELS[:5])
['s40rts', 'glad_m25', 'semucb_wm1', 'saw642an', ...]
>>>
>>> # Load S40RTS model
>>> s40rts = gdrift.SeismicModel("s40rts", nearest_neighbours=8)
>>> # Query dVs at specific location (lat, lon, depth)
>>> dvs = s40rts.at(lat=45.0, lon=10.0, depth=1000e3, quantity="dvs")
>>>
>>> # Use different interpolation kernel
>>> dvs_gauss = s40rts.at(lat=45.0, lon=10.0, depth=1000e3,
...                       quantity="dvs", kernel="gaussian")

Notes
-----
- Model names should be provided WITHOUT the "3d_seismic_" prefix
  (e.g., use "s40rts", not "3d_seismic_s40rts")
- Coordinates: lat/lon in degrees, depth in meters from surface
- Interpolation is limited to `maximum_distance` (200 km) from data points
- Below `minimum_distance` (1 m), returns exact values without interpolation

See Also
--------
gdrift.EarthModel3D : Base class with interpolation infrastructure
gdrift.load_dataset : Download and cache HDF5 datasets
"""

from .earthmodel3d import EarthModel3D
from .io import load_dataset
from .datasetnames import DATASET_REGISTRY, DatasetType

AVAILABLE_SEISMIC_MODELS = [
    ds.name.replace("3d_seismic_", "")
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

    def _load_fields(self, labels=[]):
        """
        Load fields from the seismic model dataset.

        Parameters:
        -----------
        labels : list
            Specific labels to load. If empty, loads all available fields.
        """
        raw_model = load_dataset(f"3d_seismic_{self.model_name}")

        if len(labels) > 0:
            for label in labels:
                if label not in raw_model.keys():
                    raise ValueError(f"{label} not present in tomography model: {self.model_name}")

        if "coordinates" not in labels and len(labels) > 0:
            labels += ["coordinates"]

        # Load specified fields or all fields
        if len(labels) > 0:
            # Load only specified labels plus coordinates
            for key in labels:
                if key == "coordinates":
                    self.set_coordinates(raw_model[key])
                else:
                    self.add_quantity(key, raw_model[key])
        else:
            # Load all fields
            for key in raw_model.keys():
                if key == "coordinates":
                    self.set_coordinates(raw_model[key])
                else:
                    self.add_quantity(key, raw_model[key])
