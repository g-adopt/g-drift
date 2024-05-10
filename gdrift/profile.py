import numpy
from .constants import R_earth
from .utility import compute_gravity, compute_pressure, compute_mass
from .io import load_dataset
import scipy
from abc import ABC, abstractmethod

_PREM_FILENAME = "1d_prem"
_SOLIDUS_GHELICHKHAN = "1d_solidus_Ghelichkhan_et_al_2021_GJI"


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
            depth The depth in kilometers from the surface of the Earth.

        Returns:
            float or numpy.ndarray of the quantity
        """
        # Ensure the depth is within valid range
        self._check_depth_validity(depth)
        pass

    @abstractmethod
    def min_max_depth(self):
        """
        Returns the minum maximum depths in kilometers for which this profile is defined.

        Returns:
            tuple includiong minimum and maximum depths in kilometers.
        """
        pass

    def _validate_depth(self, depth):
        """
        Check if the provided depth is within the valid range.

        Args:
            depth: The depth to check.

        Raises:
            ValueError: If the depth is outside the valid range.
        """
        min_depth, max_depth = self.min_max_depth()
        if numpy.any((depth < min_depth) | (depth > max_depth)):
            raise ValueError(f"Depth {depth} is out of the valid range ({min_depth}, {max_depth})")


class RadialEarthModel:
    """
    Composite object containing multiple radial profiles representing different Earth properties,
    such as shear wave velocity (Vs), primary wave velocity (Vp), and density.

    Attributes:
        depth_profiles (list): A dictionary of RadialProfile instances.
    """

    def __init__(self, profiles):
        """
        Initialize the RadialEarthModel with a dictionary of radial profiles instances.

        Args:
            profiles (dict of RadialProfile): Profiles for different properties, keyed by property name.
        """
        self._profiles = profiles

    def get_profile_names(self):
        return list(self._profiles.keys())

    def at_depth(self, property_name, depth):
        """
        Get the value of a property at a given depth.

        Args:
            property_name (str): The name of the property (e.g., 'Vs', 'Vp', 'Density').
            depth (float): Depth in kilometers.

        Returns:
            float: The value of the property at the given depth.
        """
        if property_name in self._profiles.keys():
            return self._profiles[property_name].at_depth(depth)
        else:
            raise ValueError(
                "Property not found in model: {}".format(property_name))


class RadialProfileSpline(ProfileAbstract):
    def __init__(self, depth: numpy.ndarray, value: numpy.ndarray, name: str = None):
        """Profiles designed for Prem
        """
        self.depth = depth
        self.value = value
        self.name = name
        self._spline = None

    def at_depth(self, depth: float | numpy.ndarray):
        self._validate_depth(depth)  # Validate depth before processing

        if self._spline is None:
            self._spline = scipy.interpolate.interp1d(
                self.depth, self.value, kind='linear')
        return self._spline(depth)

    def min_max_depth(self):
        return (self.depth.min(), self.depth.max())


class PreRefEarthModel(RadialEarthModel):
    def __init__(self):
        fi_name = _PREM_FILENAME
        profs = load_dataset(fi_name)
        prem_profiles = {}
        for name, value in profs.items():
            if name == "depth":
                continue
            prem_profiles[name] = RadialProfileSpline(
                depth=profs.get("depth"), value=value)
        super().__init__(prem_profiles)


class SolidusProfileFromFile(RadialProfileSpline):
    def __init__(name, model_name: str):
        fi_name = model_name
        profiles = load_dataset(fi_name)
        profile_name = "solidus temperature"
        super().__init__(
            depth=profiles.get("depth"),
            value=profiles.get(profile_name),
            name=profile_name)

class HirschmannSolidus(ProfileAbstract):
    nd_radial = 1000
    def __init__(self):
        # TODO: Add support for user defined reference earth model, not just PREM
        self._is_depth_converter_setup = False

    def at_depth(self, depth: float | numpy.ndarray):
        self._validate_depth(depth)  # Validate depth before processing
        if not self._is_depth_converter_setup:
            self._setup_depth_converter()
        return self._polynomial(self._depth_to_pressure(depth))

    def _setup_depth_converter(self):
        prem = PreRefEarthModel()
        radius = numpy.linspace(0., R_earth, HirschmannSolidus.nd_radial)
        dpths = R_earth - radius
        mass = compute_mass(radius, prem.at_depth("density", dpths))
        gravity = compute_gravity(radius, mass)
        pressure = compute_pressure(radius, prem.at_depth("density", dpths), gravity)
        self._depth_to_pressure = scipy.interpolate.interp1d(dpths, pressure, kind="linear")

    def _polynomial(self, pressure):
        a = -5.904
        b = 139.44
        c = 1108.08
        # compute solidus in Kelving
        return a*(pressure/1e9)**2 + b*(pressure/1e9) + c + 273.0

    def min_max_depth(self):
        return (0., 2890.e3)