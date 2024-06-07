from typing import Dict, Optional
from numbers import Number
import numpy
from .constants import R_earth, celcius2kelvin
from .utility import compute_gravity, compute_pressure, compute_mass
from .io import load_dataset
import scipy
from abc import ABC, abstractmethod

_PREM_FILENAME = "1d_prem"


class ProfileAbstract(ABC):
    """
    Abstract class representing a radial profile of a quantity within the Earth.

    This class requires subclasses to implement methods for calculating the quantity
    at a given depth and for returning the maximum depth applicable for the profile.
    """

    @abstractmethod
    def at_depth(self, depth: float | numpy.ndarray):
        """
        Retrieve the quantity (e.g., temperature, pressure, density) at a specified depth.

        Args:
            depth (float or numpy.ndarray): The depth in kilometers from the surface of the Earth.

        Returns:
            float or numpy.ndarray: The quantity at the specified depth.
        """
        # Ensure the depth is within valid range
        self._validate_depth(depth)
        pass

    @abstractmethod
    def min_max_depth(self):
        """
        Returns the minimum and maximum depths in kilometers for which this profile is defined.

        Returns:
            tuple: Including minimum and maximum depths in kilometers.
        """
        pass

    def _validate_depth(self, depth):
        """
        Check if the provided depth is within the valid range.

        Args:
            depth (float or numpy.ndarray): The depth to check.

        Raises:
            ValueError: If the depth is outside the valid range.
        """
        min_depth, max_depth = self.min_max_depth()
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

    def __init__(self, profiles: Dict[str, ProfileAbstract]):
        """
        Initialize the RadialEarthModel with a dictionary of radial profiles instances.

        Args:
            profiles (dict of RadialProfile): Profiles for different properties, keyed by property name.
        """
        if not all(isinstance(profile, ProfileAbstract) for profile in profiles.values()):
            raise ValueError(
                "All profiles must be instances of ProfileAbstract or its subclasses.")
        self._profiles = profiles

    def get_profile_names(self):
        return list(self._profiles.keys())

    def at_depth(self, property_name: str, depth: Number | numpy.ndarray):
        """
        Get the value of a property at a given depth.

        Args:
            property_name (str): The name of the property (e.g., 'Vs', 'Vp', 'Density').
            depth (float or numpy.ndarray): Depth in kilometers.

        Returns:
            float or numpy.ndarray: The value of the property at the given depth.
        """
        if property_name in self._profiles:
            return self._profiles[property_name].at_depth(depth)
        else:
            raise ValueError(f"Property not found in model: {property_name}")


class RadialProfileSpline(ProfileAbstract):
    def __init__(self, depth: numpy.ndarray, value: numpy.ndarray, name: Optional[str] = ""):
        """Initialize a radial profile, by establishing a spline for each profile.

        Args:
            depth (numpy.ndarray): Array of depths.
            value (numpy.ndarray): Array of corresponding values.
            name (str, optional): Name of the profile. Defaults to None.
        """
        self.depth = depth
        self.value = value
        self.name = name
        self._is_spline_made = False

    def at_depth(self, depth: Number | numpy.ndarray):
        """quiry a profile at certain depth[s]

        Args:
            depth (Number | numpy.ndarray): depth(s) of enquiry

        Returns:
            Number | numpy.ndarray: Value[s] of the profiles at those depths.
        """
        self._validate_depth(depth)  # Validate depth before processing

        if not self._is_spline_made:
            self._spline = scipy.interpolate.interp1d(
                self.depth, self.value, kind='linear')
            self._is_spline_made = True

        return self._spline(depth)

    def min_max_depth(self):
        """ returns minimum and maximum values of the profile
            to avoid extrapolation

        Returns:
            tuple(min, max): minimum and maximum values of the profile
        """
        return (self.depth.min(), self.depth.max())


class PreliminaryRefEarthModel(RadialEarthModel):
    def __init__(self):
        """ Preliminary Reference Earth Model (PREM)
        Dziewonski, Adam M., and Don L. Anderson. "Preliminary reference Earth model." Physics of the earth and planetary interiors 25.4 (1981): 297-356.

        The object is of type(RadialEarthModel), and can be queried at certain depths for available profiles.
        """
        fi_name = _PREM_FILENAME
        profs = load_dataset(fi_name)
        prem_profiles = {}
        for name, value in profs.items():
            if name == "depth":
                continue
            prem_profiles[name] = RadialProfileSpline(
                depth=profs.get("depth"), value=value, name=name)
        super().__init__(prem_profiles)


class SolidusProfileFromFile(RadialProfileSpline):
    """
    A class for loading radial profiles of the solidus temperature from a dataset.

    This class extends `RadialProfileSpline` to specifically handle loading,
    and utilising available profiles related to the solidus temperature in the mantle.

    Attributes:
        profile_name (str): Static attribute to hold the name of the profile
                            specifically for solidus temperature.
        model_name (str): The name of the model/dataset from which profiles are loaded.
        description (str, optional): A brief description of the profile's purpose or characteristics.
    """
    profile_name = "solidus temperature"

    def __init__(self, model_name: str, description=None):
        """
        Initialises the SolidusProfileFromFile instance by loading the solidus temperature
        profile from the specified dataset.

        Args:
            model_name (str): The name of the dataset from which to load the solidus temperature profile.
            description (str, optional): A description of the profile, which may include its use or
                                         any other relevant information.

        Raises:
            KeyError: If the necessary data fields are missing in the dataset.
        """
        self.model_name = model_name
        self.description = description

        profiles = load_dataset(self.model_name)
        super().__init__(depth=profiles.get("depth"),
                         value=profiles[SolidusProfileFromFile.profile_name],
                         name=description)


class HirschmannSolidus(ProfileAbstract):
    nd_radial = 1000
    maximum_pressure = 10e9

    def __init__(self):
        self._is_depth_converter_setup = False
        self.name = "Hirschmann 2000"

    def at_depth(self, depth: float | numpy.ndarray):
        if not self._is_depth_converter_setup:
            self._setup_depth_converter()
        self._validate_depth(depth)  # Validate depth before processing
        return self._polynomial(self._depth_to_pressure(depth))

    def _setup_depth_converter(self):
        prem = PreliminaryRefEarthModel()
        radius = numpy.linspace(0., R_earth, HirschmannSolidus.nd_radial)
        depths = R_earth - radius
        mass = compute_mass(radius, prem.at_depth("density", depths))
        gravity = compute_gravity(radius, mass)
        pressure = compute_pressure(
            radius, prem.at_depth("density", depths), gravity)
        self._depth_to_pressure = scipy.interpolate.interp1d(
            depths, pressure, kind="linear")

    def _polynomial(self, pressure):
        a = -5.904
        b = 139.44
        c = 1108.08
        # compute solidus in Kelvin
        return a * (pressure / 1e9) ** 2 + b * (pressure / 1e9) + c + celcius2kelvin

    def min_max_depth(self):
        if not self._is_depth_converter_setup:
            self._setup_depth_converter()

        def pressure_difference(depth):
            return (self._depth_to_pressure(depth)
                    - HirschmannSolidus.maximum_pressure)

        max_depth = scipy.optimize.root_scalar(
            pressure_difference, method="bisect", bracket=[0, 2000e3]).root
        return (0., max_depth)
