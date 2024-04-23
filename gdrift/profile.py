import numpy
from .io import load_dataset
import scipy
from abc import ABC, abstractmethod


class RadialProfile(ABC):
    """
    Abstract class representing a radial profile of a quantity within the Earth.

    This class requires subclasses to implement methods for calculating the quantity
    at a given depth and for returning the maximum depth applicable for the profile.

    Args:
        depth(numpy.ndarray): The depth in kilometers from the surface of the Earth.
        value(numpy.ndarray): The quantity associated with the specified depth.
    """

    def __init__(self, depth: numpy.ndarray, value: numpy.ndarray, name: str = None):
        """
      Initialize the RadialProfile with a depth and a corresponding quantity.

       Args:
            depth(float): The depth in kilometers from the surface of the Earth.
            value(float): The quantity associated with the specified depth.
            name(str): The name associated with the value
        """
        self.depth = depth
        self.value = value
        self.name = name

    @abstractmethod
    def at_depth(self, depth: float | numpy.ndarray):
        """
        Retrieve the quantity (e.g., temperature, pressure, density) at a specified depth.

        Args:
            depth The depth in kilometers from the surface of the Earth.

        Returns:
            float or numpy.ndarray of the quantity
        """
        pass

    @abstractmethod
    def min_max_depth(self):
        """
        Returns the minum maximum depths in kilometers for which this profile is defined.

        Returns:
            tuple includiong minimum and maximum depths in kilometers.
        """
        pass


class RadialEarthModel:
    """
    Composite object containing multiple radial profiles representing different Earth properties,
    such as shear wave velocity (Vs), primary wave velocity (Vp), and density.

    Attributes:
        depth_profiles (list): A dictionary of RadialProfile instances.
    """

    def __init__(self, profiles):
        """
        Initialize the RadialEarthModel with a dictionary of RadialProfile instances.

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


class PreRefEarthProf(RadialProfile):
    def __init__(self, depth: numpy.ndarray, value: numpy.ndarray, name: str = None):
        """Profiles designed for Prem
        """
        super().__init__(depth, value, name)
        self._spline = None

    def at_depth(self, depth: float | numpy.ndarray):
        if self._spline is None:
            self._spline = scipy.interpolate.interp1d(
                self.depth, self.value, kind='linear')
        return self._spline(depth)

    def min_max_depth(self):
        return (self.depth.min(), self.depth.max())


class PreRefEarthModel(RadialEarthModel):
    def __init__(self):
        fi_name = "1d_prem"
        profs = load_dataset(fi_name)
        prem_profiles = {}
        for name, value in profs.items():
            if name == "depth":
                continue
            prem_profiles[name] = PreRefEarthProf(
                depth=profs.get("depth"), value=value)
        super().__init__(prem_profiles)


def compute_mass(radius, density):
    """
    Compute the mass enclosed within each radius using the cumulative trapezoidal rule.

    Args:
        radius (numpy.ndarray): Array of radii from the center of the Earth or other celestial body.
        density (numpy.ndarray): Array of densities corresponding to each radius.

    Returns:
        numpy.ndarray: Array of cumulative mass enclosed up to each radius.
    """
    mass_enclosed = numpy.zeros_like(radius)
    for i in range(1, len(radius)):
        shell_volume = 4/3 * numpy.pi * (radius[i]**3 - radius[i-1]**3)
        average_density = (density[i] + density[i-1]) / 2
        mass_enclosed[i] = mass_enclosed[i-1] + shell_volume * average_density
    return mass_enclosed


def compute_gravity(radius, mass_enclosed):
    """
    Compute gravitational acceleration at each radius based on the enclosed mass.

    Args:
        radius (numpy.ndarray): Array of radii from the center.
        mass_enclosed (numpy.ndarray): Array of cumulative mass enclosed up to each radius.

    Returns:
        numpy.ndarray: Array of gravitational acceleration at each radius.
    """
    gravity = numpy.zeros_like(radius)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        gravity = scipy.constants.G * mass_enclosed / radius**2
        # approximate central gravity as slightly above it to avoid NaN
        gravity[0] = gravity[1]
    return gravity


def compute_pressure(radius, density, gravity):
    """
    Calculate the hydrostatic pressure at each radius based on the density and gravitational acceleration.

    Args:
        radius (numpy.ndarray): Array of radii from the center to the surface.
        density (numpy.ndarray): Array of densities at each radius.
        gravity (numpy.ndarray): Array of gravitational accelerations at each radius.

    Returns:
        numpy.ndarray: Array of pressures calculated from the surface inward to each radius.
    """
    pressure = numpy.zeros_like(radius)
    for i in range(len(radius)-2, -1, -1):
        dr = radius[i+1] - radius[i]
        avg_density = (density[i] + density[i+1]) / 2
        avg_gravity = (gravity[i] + gravity[i+1]) / 2
        pressure[i] = pressure[i+1] + avg_density * avg_gravity * dr
    return pressure
