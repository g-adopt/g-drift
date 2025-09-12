from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
from .constants import R_cmb, R_earth
from .io import load_dataset
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
    def __init__(self, nearest_neighbours=8, default_max_distance=200e3):
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

    def at(self, label: Union[str, List[str]], coordinates: np.array):
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
        # checking if the quantity is available
        self.check_quantity(label)

        if self.coordinates is None:
            raise ValueError("Coordinates not set for the model")

        # If the KDtree is not created, create it
        if self.tree is None:
            self.tree = cKDTree(self.coordinates)

        # Finding the nearest points and the indices
        distances, indices = self.tree.query(coordinates, k=self.nearest_neighbours)

        # Interpolating the values to the points
        res_dictionary = interpolate_to_points(
            create_labeled_array(self.available_fields, enlist(label)),
            distances,
            indices)

        return np.squeeze(res_dictionary)


class SeismicEarthModel(EarthModel3D):
    # Hard coding a minium distance, below which we do not interpolate
    minimum_distance = 1e-3
    # Hard coding a longest distance beyond which we don't have access to data
    maximum_distance = 200e3

    def __init__(self, model_name, labels=[]):
        super().__init__()
        self.model_name = model_name
        self._load_fields(labels=labels)
        self.tree_is_created = False

    def check_extent(self, x, y, z, tolerance=1e-3):
        radius = np.sqrt(x**2 + y**2 + z**2)

        return (all(radius >= REVEALSeismicModel3D.rmin - tolerance) and all(radius <= REVEALSeismicModel3D.rmax + tolerance))

    def _interpolate_to_points(self, label, coordinates, k=20):
        # Making sure we have a list of items
        label = enlist(label)

        # Making sure
        if label not in self.available_fields.keys():
            raise ValueError(f"{label} does not exist for model {self.model_name}")

        # generate the KDTree only if it has not been created already.
        if not self.tree_is_created:
            self.tree = cKDTree(self.coordinates)
            self.tree_is_created = True

        # finding the nearest k points
        dists, inds = self.tree.query(coordinates, k=k)

        safe_dists = np.where(dists < REVEALSeismicModel3D.minimum_distance,
                              dists, REVEALSeismicModel3D.minimum_distance)
        replace_flg = dists[:, 0] < REVEALSeismicModel3D.minimum_distance

        if len(self.available_fields[label].shape) > 1:
            ret = np.einsum("i, ik -> ik", np.sum(1 / safe_dists, axis=1), np.einsum(
                "ij, ijk -> ik", 1 / safe_dists, self.available_fields[label][inds]))
            ret[replace_flg, :] = self.available_fields[label][inds[replace_flg, 0], :]
        else:
            ret = np.einsum("ij, ij->i", 1 / safe_dists,
                            self.available_fields[label][inds]) / np.sum(1 / safe_dists, axis=1)
            ret[replace_flg] = self.available_fields[label][inds[replace_flg, 0]]
        return ret

    def _load_fields(self, labels=[]):
        data = load_dataset(self.model_name)
        if len(labels) > 0:
            for label in labels:
                if label not in data.keys():
                    raise ValueError(
                        f"{label} not present in tomography model: {self.model_name}")

        if "coordinates" not in labels:
            labels += ["coordinates"]

        for key in data.keys() if len(labels) == 1 else labels:
            self.add_quantity(key, data[key])
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

        return all(radius >= REVEALSeismicModel3D.rmin - tolerance) and all(radius <= REVEALSeismicModel3D.rmax + tolerance)

    def _interpolate_to_points(self, label, coordinates, k=8):
        if not self.tree_is_created:
            self.tree = cKDTree(self.coordinates)

        dists, inds = self.tree.query(coordinates, k=k)
        safe_dists = np.where(dists < REVEALSeismicModel3D.minimum_distance, dists, REVEALSeismicModel3D.minimum_distance)
        replace_flg = dists[:, 0] < REVEALSeismicModel3D.minimum_distance

        if len(self.available_fields[label].shape) > 1:
            ret = np.einsum("i, ik -> ik", np.sum(1 / safe_dists, axis=1), np.einsum("ij, ijk -> ik", 1 / safe_dists, self.available_fields[label][inds]))
            ret[replace_flg, :] = self.available_fields[label][inds[replace_flg, 0], :]
        else:
            ret = np.einsum("ij, ij->i", 1 / safe_dists, self.available_fields[label][inds]) / np.sum(1 / safe_dists, axis=1)
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
