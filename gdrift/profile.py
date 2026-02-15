"""One-dimensional radial Earth profiles with spline interpolation.

This module provides abstractions for working with 1D profiles of physical
quantities (density, temperature, velocity, etc.) as a function of depth
within the Earth. It supports cubic spline interpolation, composite radial
models with multiple property profiles, and specialized solidus temperature
profiles based on experimental petrology.

The primary use cases are:
- Loading reference Earth models like PREM
- Querying radial profiles at arbitrary depths
- Computing derived quantities (gravity, pressure, mass)
- Applying solidus temperature constraints in thermodynamic calculations

Class Hierarchy
---------------
AbstractProfile (ABC)
  ├── SplineProfile : Cubic spline interpolation of (depth, value) pairs
  └── HirschmannSolidusProfile : Depth → pressure → solidus temperature

RadialEarthModel : Container for multiple named profiles
  ├── RadialEarthModelFromFile : Load profiles from HDF5 datasets
  │     └── PreliminaryRefEarthModel : PREM model singleton
  └── HirschmannSolidus : Mantle solidus from experimental petrology

Key Classes
-----------
SplineProfile : 1D spline interpolation with optional extrapolation
RadialEarthModel : Multi-profile container with gravity/pressure computation
RadialEarthModelFromFile : Load HDF5 datasets as radial models
PreliminaryRefEarthModel : PREM reference model
HirschmannSolidusProfile : Experimental solidus from Hirschmann (2000)
HirschmannSolidus : Radial model wrapper for solidus profile

Examples
--------
>>> import gdrift
>>> # Load PREM and query density at 670 km depth
>>> prem = gdrift.PreliminaryRefEarthModel()
>>> rho = prem.get_profile("density")
>>> density_at_670 = rho.at_depth(670e3)  # meters
>>>
>>> # Create custom spline profile
>>> import numpy as np
>>> depths = np.linspace(0, 2890e3, 100)
>>> values = 5000 + depths / 1e6  # simple gradient
>>> profile = gdrift.SplineProfile(depths, values, name="custom_rho")
>>> value_at_500km = profile.at_depth(500e3)
>>>
>>> # Load solidus temperature profile
>>> andrault = gdrift.RadialEarthModelFromFile("1d_solidus_Andrault_et_al_2011_EPSL")
>>> solidus = andrault.get_profile("solidus temperature")
>>> T_solidus_410km = solidus.at_depth(410e3)

Notes
-----
All depth coordinates are in meters from the surface (depth increases downward).
Spline extrapolation is disabled by default to prevent unphysical values outside
the data range. Use `extrapolate=True` in SplineProfile for linear extrapolation.

See Also
--------
gdrift.constants : R_earth, R_cmb constants
gdrift.utility : compute_gravity, compute_pressure, compute_mass
"""

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
        """Retrieve the quantity at a specified depth or depths.

        Parameters
        ----------
        depth : float or ndarray
            Depth(s) in meters from the surface of the Earth. Can be a
            scalar or array.

        Returns
        -------
        float or ndarray
            The quantity value(s) at the specified depth(s). Shape matches
            the input depth array.

        Notes
        -----
        Subclasses must implement this method to define how the profile
        is evaluated at arbitrary depths (e.g., via interpolation).
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
        """Initialize a radial profile with spline interpolation.

        Creates a 1D profile by interpolating between (depth, value) pairs
        using scipy.interpolate.interp1d. The spline is constructed lazily
        on first query for efficiency.

        Parameters
        ----------
        depth : array_like
            Array of depth values in meters from the surface. Must be
            monotonically increasing or decreasing.
        value : array_like
            Array of property values corresponding to each depth. Must
            have the same length as depth.
        name : str, optional
            Name of the profile (e.g., "density", "vs", "temperature").
            Default is "Profile".
        spline_type : str, optional
            Interpolation method passed to scipy.interpolate.interp1d.
            Options: "linear", "cubic", "quadratic", etc. Default is "linear".
        extrapolate : bool, optional
            Whether to allow extrapolation outside the depth range. If False,
            queries outside the range raise ValueError. If True, uses linear
            extrapolation. Default is False.

        Notes
        -----
        The spline is not created until the first call to `at_depth()` to
        avoid unnecessary computation during initialization.
        """
        # All profiles should come with a name
        super().__init__(name)
        self.raw_depth = depth
        self.raw_value = value
        # display_name defaults to name, can be overridden for richer labels
        self.display_name = name

        self.spline_type = spline_type
        self._is_spline_made = False

        # Check if the spline should extrapolate:
        self.extrapolate = extrapolate

    def at_depth(self, depth: Number) -> Number:
        """Query the profile value at a specified depth or depths.

        Evaluates the spline at the requested depth(s). Constructs the
        spline on first call if not already created.

        Parameters
        ----------
        depth : float or ndarray
            Depth(s) in meters from the surface at which to query the
            profile. Can be a scalar or array.

        Returns
        -------
        float or ndarray
            Profile value(s) at the specified depth(s). Shape matches the
            input depth array.

        Raises
        ------
        ValueError
            If extrapolate=False and the provided depth is outside the
            valid range [min_depth, max_depth].

        Notes
        -----
        The spline is created lazily on the first call to this method using
        scipy.interpolate.interp1d with the specified spline_type.
        """
        # Make sure the query depth is within the valid range if not extrapolating
        if not self.extrapolate:
            self._validate_depth(depth)

        # If the spline has not been made, create it
        if not self._is_spline_made:
            # Create a linear spline
            self._spline = scipy.interpolate.interp1d(
                self.raw_depth, self.raw_value, kind=self.spline_type,
                bounds_error=False,
                fill_value="extrapolate")

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
        """Initialize a radial Earth model by loading profiles from an HDF5 dataset.

        Loads all property profiles from a registered dataset file. The HDF5
        file must contain a "depth" array and one or more property arrays
        (e.g., "density", "vs", "vp"). Each property is wrapped in a
        SplineProfile for interpolation.

        Parameters
        ----------
        model_name : str
            Name of the registered dataset (e.g., "1d_prem",
            "1d_solidus_Andrault_et_al_2011_EPSL"). Must exist in the
            dataset registry.
        description : str, optional
            Human-readable description of the model. If None, no description
            is set. Default is None.

        Raises
        ------
        ValueError
            If model_name is not in the dataset registry.
        KeyError
            If the HDF5 file does not contain a "depth" array.

        Examples
        --------
        >>> import gdrift
        >>> andrault = gdrift.RadialEarthModelFromFile(
        ...     "1d_solidus_Andrault_et_al_2011_EPSL")
        >>> print(andrault.get_profile_names())
        ['solidus temperature']
        >>> solidus = andrault.get_profile("solidus temperature")
        >>> T = solidus.at_depth(410e3)
        """
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
            # Keep internal key as-is, but include description in display name
            display_name = f"{name} ({self.description})" if self.description else name
            profile = SplineProfile(depth=depths, value=value, name=name, spline_type="linear")
            profile.display_name = display_name
            all_profiles.append(profile)

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
        """Initialize the Preliminary Reference Earth Model (PREM).

        Loads the PREM dataset (Dziewonski & Anderson, 1981) containing
        reference profiles for density, seismic velocities, elastic moduli,
        pressure, and gravity as a function of radius/depth.

        The model spans from Earth's center to the surface and includes
        discontinuities at major boundaries (e.g., core-mantle boundary,
        410 km, 660 km discontinuities).

        Notes
        -----
        PREM is the standard 1D reference model for seismology and geodynamics.
        It represents a spherically symmetric, non-rotating, oceanless Earth.

        Examples
        --------
        >>> import gdrift
        >>> prem = gdrift.PreliminaryRefEarthModel()
        >>> print(prem.get_profile_names())
        ['density', 'vs', 'vp', ...]
        >>> rho_profile = prem.get_profile("density")
        >>> rho_670km = rho_profile.at_depth(670e3)

        References
        ----------
        Dziewonski, A. M., & Anderson, D. L. (1981). Preliminary reference
        Earth model. Physics of the Earth and Planetary Interiors, 25(4),
        297-356. https://doi.org/10.1016/0031-9201(81)90046-7
        """
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
    _display_name = "solidus temperature (Hirschmann 2000)"

    def __init__(self):
        """Initialize the Hirschmann solidus temperature profile.

        Creates a solidus profile based on the experimental petrology data
        of Hirschmann (2000). The solidus temperature is computed as a
        quadratic function of pressure, with pressure derived from depth
        using PREM density and gravity profiles.

        The depth-to-pressure converter is initialized lazily on first query
        to avoid loading PREM during module import.

        Notes
        -----
        Valid depth range: 0 to ~730 km (pressure < 10 GPa).
        The solidus represents the temperature at which partial melting begins
        in fertile peridotite (pyrolite) composition.

        Examples
        --------
        >>> import gdrift
        >>> solidus_profile = gdrift.profile.HirschmannSolidusProfile()
        >>> T_solidus_100km = solidus_profile.at_depth(100e3)
        >>> print(f"Solidus at 100 km: {T_solidus_100km:.1f} K")

        References
        ----------
        Hirschmann, M. M. (2000). Mantle solidus: Experimental constraints
        and the effects of peridotite composition. Geochemistry, Geophysics,
        Geosystems, 1(10). https://doi.org/10.1029/2000GC000070
        """
        self._is_depth_converter_setup = False
        self.name = HirschmannSolidusProfile._name
        self.display_name = HirschmannSolidusProfile._display_name

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
        """Initialize a Hirschmann solidus radial Earth model.

        Creates a RadialEarthModel containing a single profile: the Hirschmann
        (2000) solidus temperature. This is a convenience wrapper around
        HirschmannSolidusProfile that conforms to the RadialEarthModel interface.

        The solidus can be accessed via `get_profile("solidus temperature")` or
        directly through the `solidus_profile` attribute.

        Examples
        --------
        >>> import gdrift
        >>> solidus_model = gdrift.HirschmannSolidus()
        >>> solidus_profile = solidus_model.get_profile("solidus temperature")
        >>> T_410km = solidus_profile.at_depth(410e3)
        >>> print(f"Solidus at 410 km: {T_410km:.1f} K")

        See Also
        --------
        HirschmannSolidusProfile : The underlying solidus implementation
        """
        # Initialize the solidus profile
        self.solidus_profile = HirschmannSolidusProfile()

        # Initialize the RadialEarthModel
        super().__init__(self.solidus_profile)
