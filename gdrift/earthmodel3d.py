from abc import ABC, abstractmethod
import numpy as np
from .constants import R_cmb, R_earth
from .io import load_dataset
from scipy.spatial import cKDTree


class EarthModel3D(ABC):
    def __init__(self):
        """
        Initialize the Earth model with a given extent.

        Parameters:
        extent (tuple): A tuple of tuples representing the minimum and maximum
                        coordinates ((x_min, x_max), (y_min, y_max), (z_min, z_max)).
        """
        self.coordinates = None
        self.available_fields = {}

    def add_quantity(self, label, field):
        """
        Add a quantity to the list of available quantities.

        Parameters:
        quantity (str): The name of the quantity to add.
        """
        if label == "coordinates":
            self.coordinates = field[:]
        else:
            self.available_fields[label] = field

    def check_quantity(self, quantity):
        """
        Check if a quantity is available in the model.

        Parameters:
        quantity (str): The name of the quantity to check.

        Returns:
        bool: True if the quantity is available, False otherwise.
        """
        return quantity in self.available_fields.keys()

    @abstractmethod
    def check_extent(self, x, y, z):
        """
        Check if the given coordinates are within the model's extent.

        Parameters:
        x, y, z (float): The coordinates to check.

        Returns:
        bool: True if the coordinates are within the extent, False otherwise.
        """
        pass

    def at(self, x, y, z, label):
        """
        Get the value of a quantity at the specified coordinates.

        Parameters:
        x, y, z (float): The coordinates where the value is requested.
        quantity (str): The name of the quantity to retrieve.

        Returns:
        float: The value of the quantity at the specified coordinates.

        Raises:
        ValueError: If the quantity is not available or the coordinates are out of bounds.
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)

        if not self.check_quantity(label):
            raise ValueError(f"Quantity '{label}' is not available in the model.")

        if not self.check_extent(x, y, z):
            raise ValueError("Coordinates are out of the model's extent.")
        return self._interpolate_to_points(label, np.column_stack((x, y, z)))

    @abstractmethod
    def _load_fields(self, labels=[]):
        pass


    @abstractmethod
    def _interpolate_to_points(self, label, coords):
        pass


class REVEALSeismicModel3D(EarthModel3D):
    fi_name = "REVEAL"
    rmin = R_cmb
    rmax = R_earth
    minimum_distance = 1e-3
    def __init__(self, labels=[]):
        super().__init__()
        self._load_fields(labels=labels)
        self.tree_is_created = False

    def check_extent(self, x, y, z, tolerance=1e-3):
        radius = np.sqrt(x**2 + y**2 + z**2)

        return (all( radius >= REVEALSeismicModel3D.rmin - tolerance)
            and all( radius <= REVEALSeismicModel3D.rmax + tolerance))

    def _interpolate_to_points(self, label, coordinates, k=8):
        if not self.tree_is_created:
            self.tree = cKDTree(self.coordinates)

        dists, inds = self.tree.query(coordinates, k=k)
        safe_dists = np.where(dists<REVEALSeismicModel3D.minimum_distance, dists, REVEALSeismicModel3D.minimum_distance)
        replace_flg = dists[:, 0] < REVEALSeismicModel3D.minimum_distance

        if len(self.available_fields[label].shape) > 1:
            ret = np.einsum("i, ik -> ik", np.sum(1/safe_dists, axis=1), np.einsum("ij, ijk -> ik", 1/safe_dists, self.available_fields[label][inds]))
            ret[replace_flg, :] = self.available_fields[label][inds[replace_flg, 0], :]
        else:
            ret = np.einsum("ij, ij->i", 1/safe_dists, self.available_fields[label][inds])/ np.sum(1/safe_dists, axis=1)
            ret[replace_flg] = self.available_fields[label][inds[replace_flg, 0]]
        return ret


    def _load_fields(self, labels=[]):
        reveal_data = load_dataset(REVEALSeismicModel3D.fi_name)

        if len(labels) > 0:
            for label in labels:
                if label not in reveal_data.keys():
                    raise ValueError(f"{label} not present in REVEAL")

        if "coordinates" not in labels:
            labels += ["coordinates"]

        for key in reveal_data.keys() if len(labels) == 1 else labels:
            self.add_quantity(key, reveal_data[key])