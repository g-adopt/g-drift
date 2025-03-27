from typing import Optional, List, Union
from numbers import Number
from abc import ABC, abstractmethod
from .constants import R_earth, celcius2kelvin
from .utility import compute_gravity, compute_pressure, compute_mass, enlist
from .io import load_dataset
import scipy
import numpy


class AbstractProfile(ABC):
    """
    Abstract class representing a radial profile of a quantity within the Earth.

    This class requires subclasses to implement methods for calculating the quantity
    at a given depth and for returning the maximum depth applicable for the profile.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def at_depth(self, depth: Number) -> Number:
        """
        Retrieve the quantity (e.g., temperature, pressure, density) at a specified depth.

        Args:
            depth (float or numpy.ndarray): The depth~(SI unit) from the surface of the Earth.

        Returns:
            float or numpy.ndarray: The quantity at the specified depth.
        """
        pass


class SplineProfile(AbstractProfile):
    """
    A class to represent a spline profile.

    Attributes:
        raw_depth (Number): Array of depths.
        raw_value (Number): Array of corresponding values.
        spline_type (str): Type of spline to use. Defaults to "linear".
        _is_spline_made (bool): Flag to indicate if the spline has been created.
    """

    def __init__(self, depth: Number, value: Number, name: Optional[str] = "Profile", spline_type: str = "linear", extrapolate: bool = False):
        """
        Initialise a radial profile by establishing a spline.

            depth (Number): Array of depths.
            value (Number): Array of corresponding values.
            name (Optional[str], optional): Name of the profile. Defaults to an empty string.
            spline_type (str, optional): Type of spline to use. Defaults to "linear".
            extrapolate (bool, optional): Whether to allow extrapolation. Defaults to False.
        """
        # All profiles should come with a name
        super().__init__(name)
        self.raw_depth = depth
        self.raw_value = value

        self.spline_type = spline_type
        self._is_spline_made = False

        # Check if the spline should extrapolate:
        self.extrapolate = extrapolate

    def at_depth(self, depth: Number) -> Number:
        """
        Query the profile value at a specified depth or depths.

            depth (Number): A single depth value or an array of depth values
                            at which to query the profile.

            Number or numpy.ndarray: The profile value(s) at the specified depth(s).
                                     Returns a single value if a single depth is provided,
                                     or an array of values if an array of depths is provided.

        Raises:
            ValueError: If the provided depth is out of the valid range.
        """
        # Make sure the query depth is within the valid range if not extrapolating
        if not self.extrapolate:
            self._validate_depth(depth)

        # If the spline has not been made, create it
        if not self._is_spline_made:
            # Create a linear spline
            self._spline = scipy.interpolate.interp1d(
                self.raw_depth, self.raw_value, kind=self.spline_type,
                bounds_error=False if self.extrapolate is True else False,
                fill_value="extrapolate")

            self._is_spline_made = True

        # Query the spline
        return self._spline(depth)

        # If the spline has not been made, create it
        if not self._is_spline_made:
            # Create a linear spline
            self._spline = scipy.interpolate.interp1d(self.raw_depth, self.raw_value, kind=self.spline_type)
            self._is_spline_made = True

        # Query the spline
        return self._spline(depth)

    def min_max_depth(self):
        """
        Calculate the minimum and maximum depth values of the profile to prevent extrapolation.

            tuple: A tuple containing the minimum and maximum depth values (min, max).
        """
        return (self.raw_depth.min(), self.raw_depth.max())

    def _validate_depth(self, depth: Number):
        """
        Check if the provided depth is within the valid range.

        Args:
            depth (float or numpy.ndarray): The depth to check.

        Raises:
            ValueError: If the depth is outside the valid range.
        """
        # Get the min and max depth
        min_depth, max_depth = self.min_max_depth()

        # Check if the depth is within the valid range
        if numpy.any((depth < min_depth) | (depth > max_depth)):
            raise ValueError(
                f"Depth {depth} is out of the valid range ({min_depth}, {max_depth})")


class RadialEarthModel:
    """
    Class representing reference Earth Models such as PREM or AK135
    Composite object containing multiple radial profiles representing different Earth properties,
    such as shear wave velocity (Vs), primary wave velocity (Vp), and density. T

    Attributes:
        depth_profiles (dict): A dictionary of RadialProfile instances.
    """

    def __init__(self, profiles: Union[AbstractProfile, List[AbstractProfile]]):
        """
        Initialise the RadialEarthModel with a dictionary of radial profiles instances.

        Args:
            profiles (dict of RadialProfile): Profiles for different properties, keyed by property name.
        """
        # Make sure we have a list
        profiles = enlist(profiles)

        # Store the profiles in a dictionary
        self._profiles = {}
        # Add the profiles to the dictionary
        for p in profiles:
            self._profiles[p.name] = p

    def get_profile(self, property_name: str) -> AbstractProfile:
        """
        Retrieve a profile by its name.

        Args:
            property_name (str): The name of the property profile to retrieve.

        Returns:
            AbstractProfile: The profile corresponding to the specified property name.

        Raises:
            ValueError: If the specified property name does not exist in the model.
        """
        # Check if the property exists in the model
        if property_name in self.get_profile_names():
            # Return the profile
            return self._profiles[property_name]
        else:
            # Raise an error if the property does not exist
            raise ValueError(f"Property {property_name} not found. Existing properties: {', '.join(self.get_profile_names())}")

    def get_profile_names(self):
        """
        Retrieve the names of all profiles.

            list: A list containing the names of all profiles.
        """
        return list(self._profiles.keys())

    def at_depth(self, property_name: str, depth: Number) -> Number:
        """
        Retrieve the value of a specified property at a given depth.
            property_name (str): The name of the property to retrieve (e.g., 'Vs', 'Vp', 'Density').
            depth (float or numpy.ndarray): The depth in kilometers at which to retrieve the property value.
            float or numpy.ndarray: The value of the specified property at the given depth.
        Raises:
            ValueError: If the specified property name does not exist in the model.

        """
        return self.get_profile(property_name).at_depth(depth)

    def min_max_depth(self, property_name: str) -> tuple:
        """
        Retrieve the minimum and maximum depth for a specified property profile.
        """
        #
        if property_name in self.get_profile_names():
            return self.get_profile(property_name).min_max_depth()


class RadialEarthModelFromFile(RadialEarthModel):
    """
    A class for loading radial profiles from a dataset.

    This class extends `SplineProfile` to specifically handle loading,
    and utilizing available profiles related to profiles in the mantle.

    Attributes:
        model_name (str): The name of the model/dataset from which profiles are loaded.
        description (str, optional): A brief description of the profile's purpose or characteristics.
    """

    def __init__(self, model_name: str, description: str = None):
        # Set the profile name
        self.model_name = model_name
        # Set the description
        self.description = description

        # Load the dataset
        profiles = load_dataset(self.model_name)

        # Get the depth
        depths = profiles.get("depth")

        # Extract the profiles as Profile objects
        all_profiles = []
        for name, value in profiles.items():
            if name == "depth":
                continue
            all_profiles.append(SplineProfile(depth=depths, value=value, name=name, spline_type="linear"))

        # Initialize the RadialEarthModel
        super().__init__(all_profiles)


class PreliminaryRefEarthModel(RadialEarthModelFromFile):
    """
    Initialises the Preliminary Reference Earth Model (PREM).
    This model is based on the work by Dziewonski and Anderson (1981) and provides a reference Earth model that can be queried at specific depths for various profiles.

    References:
        Dziewonski, Adam M., and Don L. Anderson. "Preliminary reference Earth model." Physics of the Earth and Planetary Interiors 25.4 (1981): 297-356.

    The object is of type RadialEarthModel and is initialized by loading profiles from an existing dataset. Each profile is represented as a SplineProfile object.

    Attributes:
        prem_profiles (list): A list of SplineProfile objects representing different profiles in the PREM dataset.
    """
    # Filename containing PREM property profiles
    PREM_FILENAME = "1d_prem"

    def __init__(self):
        # Initialize the RadialEarthModel
        super().__init__(PreliminaryRefEarthModel.PREM_FILENAME, "Preliminary Reference Earth Model")


class HirschmannSolidusProfile(AbstractProfile):
    """
    HirschmannSolidusProfile is the solidus model based on the work of Hirschmann (2000).

    Attributes:
        nd_radial (int): Number of radial points for interpolation.
        maximum_pressure (float): Maximum pressure in Pascals.
        name (str): Name of the profile.

    Methods:
        at_depth(depth: float | numpy.ndarray):
            Computes the solidus temperature at a given depth or array of depths.

        min_max_depth():
            Computes the minimum and maximum depths for which the pressure does not exceed the maximum pressure.
    """
    _nd_radial = 1000
    _maximum_pressure = 10e9
    _name = "solidus temperature"

    def __init__(self):
        self._is_depth_converter_setup = False
        self.name = HirschmannSolidusProfile._name

    def at_depth(self, depth: float | numpy.ndarray):
        # Setup the depth converter if not already done
        if not self._is_depth_converter_setup:
            self._setup_depth_converter()
        # Validate depth before processing
        self._validate_depth(depth)
        # Compute the solidus temperature
        return self._polynomial(self._depth_to_pressure(depth))

    # This method is used to setup the depth to pressure converter
    def _setup_depth_converter(self):
        # We use PREM to compute mass, gravity, and pressure profiles
        prem = PreliminaryRefEarthModel()
        # Compute mass, gravity, and pressure
        radius = numpy.linspace(0., R_earth, HirschmannSolidusProfile._nd_radial)
        # Compute depths
        depths = R_earth - radius
        # Compute mass, gravity, and pressure
        mass = compute_mass(radius, prem.at_depth("density", depths))
        gravity = compute_gravity(radius, mass)
        pressure = compute_pressure(
            radius, prem.at_depth("density", depths), gravity)

        # Interpolate pressure
        self._depth_to_pressure = scipy.interpolate.interp1d(
            depths, pressure, kind="linear")

    def _polynomial(self, pressure: Number) -> Number:
        """
        Computes the solidus temperature in Kelvin as a polynomial function of pressure.

        As given by Hirschmann (2000).
            pressure (Number): Pressure in Pascals.
        """
        a = -5.904
        b = 139.44
        c = 1108.08

        # compute solidus in Kelvin
        return a * (pressure / 1e9) ** 2 + b * (pressure / 1e9) + c + celcius2kelvin

    def min_max_depth(self):

        if not self._is_depth_converter_setup:
            self._setup_depth_converter()

        def pressure_difference(depth):
            return (self._depth_to_pressure(depth) - HirschmannSolidusProfile._maximum_pressure)
        max_depth = scipy.optimize.root_scalar(
            pressure_difference, method="bisect", bracket=[0, 2000e3]).root
        return (0., max_depth)

    def _validate_depth(self, depth: Number):
        """
        Check if the provided depth is within the valid range.

        Args:
            depth (Number): The depth to check.

        Raises:
            ValueError: If the depth is outside the valid range.
        """
        min_depth, max_depth = self.min_max_depth()
        if numpy.any((depth < min_depth) | (depth > max_depth)):
            raise ValueError(
                f"Depth {depth} is out of the valid range ({min_depth}, {max_depth})")


class HirschmannSolidus(RadialEarthModel):
    """
    A class representing the solidus model based on the work of Hirschmann (2000).

    Attributes:
        solidus_profile (HirschmannSolidusProfile): The solidus profile based on Hirschmann (2000).
    """

    def __init__(self):
        # Initialize the solidus profile
        self.solidus_profile = HirschmannSolidusProfile()

        # Initialize the RadialEarthModel
        super().__init__(self.solidus_profile)
