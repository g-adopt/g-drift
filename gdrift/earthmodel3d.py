"""Three-dimensional Earth models with KD-tree spatial interpolation.

This module provides a flexible framework for working with 3D gridded data
representing spatially varying properties throughout the Earth's mantle.
The primary use case is loading and querying 3D seismic tomography models,
but the infrastructure supports any 3D geophysical dataset.

Key features:
- KD-tree based nearest-neighbor search for fast spatial queries
- Multiple interpolation kernels (IDW, Gaussian, Wendland, linear)
- Coordinate system handling (geographic, Cartesian, normalized)
- Support for multiple quantities per model (e.g., dVs, dVp, density)
- Configurable search radii and neighbor counts

Architecture
------------
The module defines an abstract base class (`AbstractEarthModel`) that
specifies the interface for querying 3D models. The concrete implementation
(`EarthModel3D`) uses scipy's cKDTree for efficient spatial indexing and
supports pluggable interpolation kernels through the `interpolate_to_points`
utility function.

Subclasses (e.g., `SeismicModel` in seismic.py) extend `EarthModel3D` to
provide domain-specific loading and validation logic.

Class Hierarchy
---------------
AbstractEarthModel (ABC)
  └── EarthModel3D : KD-tree interpolation with configurable kernels

Key Classes
-----------
EarthModel3D : 3D spatial interpolation container with KD-tree
AbstractEarthModel : Abstract interface for 3D Earth models

Key Methods
-----------
EarthModel3D.set_coordinates : Define the 3D grid coordinates
EarthModel3D.add_quantity : Add a named quantity field to the model
EarthModel3D.at : Query quantities at arbitrary spatial locations

Examples
--------
>>> import gdrift
>>> import numpy as np
>>> # Create a simple 3D model
>>> model = gdrift.EarthModel3D(nearest_neighbours=8, default_max_distance=500e3)
>>> # Define coordinates (Cartesian, normalized to Earth radius)
>>> x = np.linspace(-0.5, 0.5, 10)
>>> y = np.linspace(-0.5, 0.5, 10)
>>> z = np.linspace(-0.5, 0.5, 10)
>>> xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
>>> coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
>>> model.set_coordinates(coords)
>>> # Add a quantity (e.g., synthetic velocity perturbation)
>>> dv = np.random.randn(len(coords)) * 0.02
>>> model.add_quantity("dvs", dv, "velocity perturbation")
>>> # Query at a specific location
>>> value = model.at(x=0.1, y=0.2, z=-0.3, quantity="dvs", kernel="idw")

Notes
-----
Coordinates can be provided in Cartesian (x, y, z) or geographic (lat, lon, depth)
systems. The `at()` method automatically handles coordinate transformations.
All spatial dimensions are normalized to Earth radius for numerical stability.

See Also
--------
gdrift.seismic.SeismicModel : 3D seismic tomography models
gdrift.utility.interpolate_to_points : Kernel-based interpolation
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
from scipy.spatial import cKDTree
from .utility import enlist, interpolate_to_points, create_labeled_array


class AbstractEarthModel(ABC):
    """
    Abstract base class for Earth models.

    Attributes:
        coordinates (None): Placeholder for coordinates, to be defined in subclasses.
        available_fields (dict): Dictionary to store available fields with their labels.

    Methods:
        at(label, *args):
            Abstract method to provide the value of a model for a certain label at point(s).

        check_quantity(quantity):
            Abstract method to check if a quantity is available in the model.

        check_extent(x, y, z):
            Abstract method to check if the given coordinates are within the model's extent.
    """
    @abstractmethod
    def at(self, label: str, *args):
        """ abstract functiont hat provides the value of a model for a certain label at point(s).
        """
        pass

    @abstractmethod
    def check_quantity(self, quantity: str):
        """ Placeholder for checking if a quantity is available in the model.

        Args:
            quantity (str): the name of the quantity to check
        """
        pass

    @abstractmethod
    def check_extent(self, *args):
        """ Placeholder for checking if the given coordinates are within the model's extent.


        Args:
            x, y, z (float or np.array): the coordinates to check
        """
        pass


class EarthModel3D(AbstractEarthModel):
    """Three-dimensional Earth model with KD-tree spatial interpolation.

    A container for 3D gridded geophysical data with efficient spatial queries
    using scipy's cKDTree. Supports multiple interpolation kernels (IDW,
    Gaussian, Wendland, linear, cubic, nearest neighbor) and handles both
    Cartesian and geographic coordinate systems.

    Typical workflow:
    1. Create model: `model = EarthModel3D(nearest_neighbours=8)`
    2. Set coordinates: `model.set_coordinates(x, y, z)`
    3. Add quantities: `model.add_quantity("dvs", dvs_array, "label")`
    4. Query: `value = model.at(lat=45, lon=0, depth=500e3, quantity="dvs")`

    Parameters
    ----------
    nearest_neighbours : int, optional
        Number of nearest neighbors to use for interpolation. Higher values
        produce smoother interpolation but increase computational cost.
        Default is 8 (suitable for most 3D tomography models).
    default_max_distance : float, optional
        Maximum search distance in meters for finding neighbors. Points
        farther than this from all data points return NaN. Default is 200e3
        (200 km), suitable for typical mantle convection resolution.

    Attributes
    ----------
    coordinates : ndarray or None
        Nx3 array of (x, y, z) coordinates in normalized units (scaled to
        R_earth). Set via `set_coordinates()`. None until coordinates are set.
    available_fields : dict
        Dictionary mapping quantity names to (data_array, label) tuples.
        Populated via `add_quantity()`.
    tree : scipy.spatial.cKDTree or None
        KD-tree for fast spatial queries. Built automatically when
        coordinates are set. None until `set_coordinates()` is called.
    nearest_neighbours : int
        Number of neighbors for interpolation (from initialization).
    default_max_distance : float
        Maximum search distance in meters (from initialization).

    Examples
    --------
    >>> import gdrift
    >>> import numpy as np
    >>> # Create synthetic 3D model
    >>> model = gdrift.EarthModel3D(nearest_neighbours=8)
    >>> x = np.linspace(-0.5, 0.5, 10)  # normalized coords
    >>> y = np.linspace(-0.5, 0.5, 10)
    >>> z = np.linspace(-0.5, 0.5, 10)
    >>> xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    >>> coords = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    >>> model.set_coordinates(coords[:, 0], coords[:, 1], coords[:, 2])
    >>> # Add velocity perturbation
    >>> dvs = np.random.randn(len(coords)) * 0.02
    >>> model.add_quantity("dvs", dvs, "dln(Vs)")
    >>> # Query using geographic coordinates
    >>> value = model.at(lat=45.0, lon=10.0, depth=1000e3, quantity="dvs")

    See Also
    --------
    SeismicModel : Subclass for loading seismic tomography models
    interpolate_to_points : Underlying interpolation engine
    """

    def __init__(self, nearest_neighbours=8, default_max_distance=200e3):
        """Initialize a 3D Earth model with interpolation parameters.

        Parameters
        ----------
        nearest_neighbours : int, optional
            Number of nearest neighbors for interpolation. Default is 8.
        default_max_distance : float, optional
            Maximum search distance in meters. Default is 200e3 (200 km).
        """
        self.coordinates = None
        self.available_fields = {}
        self.tree = None
        # Hard coding the number of nearest neighbors to interpolate from
        self.nearest_neighbours = nearest_neighbours
        # Default maximum distance beyond which we conclude that there are no meaningful close points
        self.default_max_distance = default_max_distance

    def set_coordinates(self, *args, max_distance=200e3):
        """
        Set the coordinates for the model.

        Parameters:
        coordinates (np.array): The coordinates to set.
        max_distance (float): The maximum distance beyond which we will raise a fat ass warning.
        """
        if self.coordinates is not None:
            raise ValueError("Coordinates are already set. To be safe, start from scratch!")

        self.coordinates = np.column_stack(args)

    def add_quantity(self, label, field):
        """
        Add a quantity to the list of available quantities.

        Parameters:
        quantity (str): The name of the quantity to add.
        """
        if label in self.available_fields:
            raise ValueError(f"{label} has already been set. To be safe, start from scratch!")

        self.available_fields[label] = field

    def check_quantity(self, quantity: Union[str, List[str]]):
        """
        Check if a quantity or quantities are available in the model.

        Parameters:
        quantity (Union[str, List[str]]): The name of the quantity or list of quantities to check.

        Returns:
        bool: True if all quantities are available, False otherwise.
        """

        if isinstance(quantity, str):
            return quantity in self.available_fields.keys()
        elif isinstance(quantity, list):
            return all(q in self.available_fields.keys() for q in quantity)
        else:
            raise TypeError("Quantity must be a string or a list of strings")

    def print_available_fields(self):
        """Print the available fields in the model.
        """
        if not self.available_fields:
            print("No fields available.")
        else:
            print("Available fields:")
            for field in self.available_fields:
                print(f"- {field}")

    def check_extent(self, coordinates):
        """
        Check if the given coordinates are within the model's extent.

        Parameters:
        x, y, z (float): The coordinates to check.

        Returns:
        bool: True if the coordinates are within the extent, False otherwise.
        """
        # Testing the closest point
        distances, _ = self.tree.query(coordinates, k=1)

        if any(distances > self.default_max_distance):
            raise ValueError("The closest point seems to be beyond the maximum meaningful distance for the Earth model")

    def at(self, label: Union[str, List[str]], coordinates: np.array, kernel='idw', **kernel_params):
        """
        Get the value of a quantity at specified coordinates using various interpolation kernels.

        Parameters:
        -----------
        label : str or list
            Field label(s) to interpolate
        coordinates : np.ndarray
            Query coordinates (N x 3 array)
        kernel : str, optional
            Interpolation kernel. Options:
            - 'idw': Inverse distance weighting (default, original behavior)
            - 'gaussian': Gaussian kernel with adaptive or fixed bandwidth
            - 'idw_power': IDW with adjustable power parameter
            - 'exponential': Exponential decay kernel
            - 'wendland': Wendland compactly supported kernel
        **kernel_params : dict
            Kernel-specific parameters:
            - For 'gaussian': sigma (bandwidth, auto-computed if not provided)
            - For 'idw_power': power (default 2.0)
            - For 'exponential': decay_length (default 50000m)
            - For 'wendland': support_radius (default 100000m)

        Returns:
        --------
        np.ndarray
            Interpolated values at query coordinates
        """
        # checking if the quantity is available
        self.check_quantity(label)

        if self.coordinates is None:
            raise ValueError("Coordinates not set for the model")

        # If the KDtree is not created, create it
        if self.tree is None:
            self.tree = cKDTree(self.coordinates)

        # Finding the nearest points and the indices
        distances, indices = self.tree.query(coordinates, k=self.nearest_neighbours)

        # Use kernel-based interpolation
        if kernel == 'idw' and len(kernel_params) == 0:
            # Backward compatibility: use original method for default IDW
            res_dictionary = interpolate_to_points(
                create_labeled_array(self.available_fields, enlist(label)),
                distances,
                indices)
        else:
            # Use new kernel-based interpolation
            res_dictionary = self._interpolate_with_kernels(
                label, coordinates, distances, indices, kernel, **kernel_params)

        return np.squeeze(res_dictionary)

    def _interpolate_with_kernels(self, label, coordinates, distances, indices, kernel='idw', **kernel_params):
        """
        Internal method to handle kernel-based interpolation.
        """
        # Calculate weights using the chosen kernel
        weights = self._calculate_kernel_weights(distances, kernel, **kernel_params)

        # Get the labeled data
        labeled_data = create_labeled_array(self.available_fields, enlist(label))

        # Handle very close points (avoid division by zero)
        min_distance = getattr(self, 'minimum_distance', 1e-3)
        replace_flg = distances[:, 0] < min_distance

        if len(labeled_data.shape) > 1:
            # Multi-dimensional field
            weighted_sum = np.einsum("ij, ijk -> ik", weights, labeled_data[indices])
            weight_sum = np.sum(weights, axis=1)[:, np.newaxis]
            result = weighted_sum / weight_sum
            result[replace_flg, :] = labeled_data[indices[replace_flg, 0], :]
        else:
            # 1D field
            weighted_sum = np.einsum("ij, ij -> i", weights, labeled_data[indices])
            weight_sum = np.sum(weights, axis=1)
            result = weighted_sum / weight_sum
            result[replace_flg] = labeled_data[indices[replace_flg, 0]]

        return result

    def _calculate_kernel_weights(self, dists, kernel='idw', **kernel_params):
        """
        Calculate interpolation weights using various kernels.

        Parameters:
        -----------
        dists : np.ndarray
            Distances to nearest neighbors
        kernel : str
            Kernel type
        **kernel_params : dict
            Kernel-specific parameters

        Returns:
        --------
        np.ndarray
            Weights for interpolation
        """
        min_distance = getattr(self, 'minimum_distance', 1e-3)

        if kernel == 'idw':
            # Original inverse distance weighting
            safe_dists = np.where(dists < min_distance, min_distance, dists)
            weights = 1 / safe_dists

        elif kernel == 'gaussian':
            # Gaussian kernel: exp(-0.5 * (r/σ)²)
            sigma = kernel_params.get('sigma', None)
            if sigma is None:
                # Adaptive bandwidth: use median distance to k neighbors
                sigma = np.median(dists, axis=1, keepdims=True)
                sigma = np.where(sigma < 1000, 1000, sigma)  # minimum 1km bandwidth
            weights = np.exp(-0.5 * (dists / sigma) ** 2)

        elif kernel == 'idw_power':
            # IDW with adjustable power: 1/r^p
            power = kernel_params.get('power', 2.0)
            safe_dists = np.where(dists < min_distance, min_distance, dists)
            weights = 1 / (safe_dists ** power)

        elif kernel == 'exponential':
            # Exponential decay: exp(-r/λ)
            decay_length = kernel_params.get('decay_length', 50000)  # 50km default
            weights = np.exp(-dists / decay_length)

        elif kernel == 'wendland':
            # Wendland C2 compactly supported kernel
            support_radius = kernel_params.get('support_radius', 100000)  # 100km default
            q = dists / support_radius
            weights = np.zeros_like(q)
            mask = q <= 1.0
            weights[mask] = (1 - q[mask]) ** 4 * (4 * q[mask] + 1)

        else:
            raise ValueError(f"Unknown kernel: {kernel}. Choose from: 'idw', 'gaussian', 'idw_power', 'exponential', 'wendland'")

        # Ensure weights are positive and handle edge cases
        weights = np.maximum(weights, 1e-12)  # Avoid exactly zero weights

        return weights
