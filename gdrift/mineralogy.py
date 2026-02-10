"""Thermodynamic lookup tables for mineral physics properties.

This module provides 2D lookup tables (depth × temperature) for mantle
mineralogy computed using self-consistent thermodynamic databases. The
primary use case is converting between temperature and seismic velocities
in forward and inverse modeling of mantle convection.

Key capabilities:
- Load pre-computed thermodynamic tables (SLB_16, SLB_21 pyrolite/basalt)
- Query properties at arbitrary (depth, temperature) pairs via bivariate spline
- Inverse lookups: velocity → temperature at fixed depth
- Compute derived properties (Vs, Vp) from elastic moduli and density
- Regularize phase transition discontinuities for smooth gradients

Architecture
------------
The module uses 2D bivariate splines (scipy.interpolate.RectBivariateSpline)
for fast interpolation of pre-tabulated mineral physics data. Tables are
stored in HDF5 format with multiple property keys (rho, bulk_mod, shear_mod,
v_s, v_p, etc.) loaded simultaneously.

Phase transitions (e.g., 410 km, 660 km) create sharp discontinuities in
properties. The `regularise_thermodynamic_table` function smooths these
discontinuities within specified depth ranges to enable gradient-based
inverse methods.

Key Classes
-----------
Table : Simple 2D table container (x, y, values)
ThermodynamicModel : 2D lookup table with forward/inverse queries
RegularisedThermodynamicModel : Dynamically created class with smoothed transitions

Key Functions
-------------
compute_swave_speed : Calculate Vs from shear modulus and density
compute_pwave_speed : Calculate Vp from bulk/shear moduli and density
regularise_thermodynamic_table : Smooth phase transition discontinuities
LinearRectBivariateSpline : Linear (kx=1, ky=1) bivariate spline factory

Examples
--------
>>> import gdrift
>>> # Load SLB 2021 pyrolite model
>>> tm = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
>>> # Query Vs at 1600 K, 500 km depth
>>> vs = tm.temperature_to_vs(temperature=1600, depth=500e3)
>>> print(f"Vs = {vs:.3f} m/s")
>>>
>>> # Inverse: convert Vs to temperature
>>> T = tm.vs_to_temperature(vs=4500, depth=500e3)
>>> print(f"Temperature = {T:.1f} K")
>>>
>>> # Query arbitrary property (if available in table)
>>> rho = tm.temperature_to_property("rho", temperature=1600, depth=500e3)
>>>
>>> # Regularize phase transitions for smooth gradients
>>> tm_smooth = gdrift.regularise_thermodynamic_table(
...     tm, regular_range={"v_s": (350e3, 750e3)})
>>> # Now vs_to_temperature uses smoothed table for inverse

Notes
-----
- All depths are in meters from the surface
- Temperatures are in Kelvin (use constants.celcius2kelvin for conversion)
- Velocities are in m/s, densities in kg/m³, moduli in Pa
- SLB_16 uses Stixrude & Lithgow-Bertelloni (2011) database
- SLB_21 uses updated parameters from Stixrude & Lithgow-Bertelloni (2021)
- Not all model/composition combinations are available (see MODELS_AVAIL,
  COMPOSITIONS_AVAIL)

See Also
--------
gdrift.anelasticity : Apply frequency-dependent corrections to velocities
gdrift.profile : 1D radial profiles for reference models
"""

import numpy
from .profile import AbstractProfile
from .io import load_dataset
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from numbers import Number
from typing import Optional, Tuple, Union, Dict
import numpy as np

# Default regular range for gradients
# This will be used in regularise_thermodynamic_table
# if nothing is provided
default_regular_range = {
    "v_s": (-np.inf, 0.0),
    "v_p": (-np.inf, 0.0),
    "rho": (-np.inf, 0.0),
}


MODELS_AVAIL = ['SLB_16', "SLB_21"]
COMPOSITIONS_AVAIL = ['pyrolite', 'basalt']


def LinearRectBivariateSpline(x, y, z):

    # This should be the case by default, but for some reason scipy does not catch this
    if not x.size == z.shape[0]:
        raise TypeError('x dimension of z must have same number of '
                        'elements as x')
    if not y.size == z.shape[1]:
        raise TypeError('y dimension of z must have same number of '
                        'elements as y')

    return RectBivariateSpline(
        x, y, z,
        bbox=[x[0], x[-1], y[0], y[-1]],
        kx=1, ky=1)


def dataset_name(model: str, composition: str):
    return f"{model}_{composition}"


class Table:
    """Base class for a 2D table with rows and columns.

    A simple container for 2D gridded data with x (row) and y (column)
    coordinates. Used internally by ThermodynamicModel to store individual
    property tables (e.g., density, velocity) as a function of depth and
    temperature.

    Parameters
    ----------
    x : array_like
        Row coordinates (typically depth values in meters).
    y : array_like
        Column coordinates (typically temperature values in Kelvin).
    vals : ndarray
        2D array of property values with shape (len(x), len(y)).
    name : str, optional
        Name of the property stored in this table (e.g., "rho", "v_s").
        Default is None.

    Attributes
    ----------
    _x : ndarray
        Row coordinates.
    _y : ndarray
        Column coordinates.
    _vals : ndarray
        2D array of property values.
    _name : str or None
        Name of the property.
    """

    def __init__(self, x, y, vals, name=None):
        """Initialize a 2D table with coordinates and values.

        Parameters
        ----------
        x : array_like
            Row coordinates (depth).
        y : array_like
            Column coordinates (temperature).
        vals : ndarray
            2D array of property values.
        name : str, optional
            Name of the property. Default is None.
        """
        self._x = x
        self._y = y
        self._vals = vals
        self._name = name

    def get_x(self):
        """Get the row coordinates (typically depth).

        Returns
        -------
        ndarray
            Row coordinates array.
        """
        return self._x

    def get_y(self):
        """Get the column coordinates (typically temperature).

        Returns
        -------
        ndarray
            Column coordinates array.
        """
        return self._y

    def get_vals(self):
        """Get the 2D array of property values.

        Returns
        -------
        ndarray
            2D array of property values with shape (len(x), len(y)).
        """
        return self._vals

    def get_name(self):
        """Get the name of the property stored in this table.

        Returns
        -------
        str or None
            Property name (e.g., "rho", "v_s"), or None if not set.
        """
        return self._name


class ThermodynamicModel(object):
    """Thermodynamic lookup table for mantle mineral physics properties.

    Provides 2D interpolation of pre-computed mineral physics properties
    (density, seismic velocities, elastic moduli) as a function of depth
    and temperature. Tables are based on self-consistent thermodynamic
    databases (Stixrude & Lithgow-Bertelloni 2011, 2021) and stored in
    HDF5 format with bivariate spline interpolation.

    The model supports:
    - Forward queries: (depth, temperature) → property (e.g., Vs, Vp, rho)
    - Inverse queries: (depth, velocity) → temperature
    - Multiple compositions (pyrolite, basalt) and database versions (SLB_16, SLB_21)
    - On-the-fly computation of Vs/Vp from elastic moduli

    Parameters
    ----------
    model : str
        Thermodynamic database version. Must be one of:
        - "SLB_16": Stixrude & Lithgow-Bertelloni (2011) parameters
        - "SLB_21": Stixrude & Lithgow-Bertelloni (2021) updated parameters
    composition : str
        Mantle composition. Available options depend on the model:
        - "pyrolite": Fertile peridotite (most common)
        - "pyroliteCFMAS": SLB_21 pyrolite in CFMAS system
        - "pyroliteNCMAS": SLB_21 pyrolite in NCMAS system
        - "basalt": MORB-like composition (SLB_16 only)
    temps : array_like, optional
        Temperature grid for subsampling (Kelvin). If None, uses full
        temperature range from dataset. Default is None.
    depths : array_like, optional
        Depth grid for subsampling (meters). If None, uses full depth
        range from dataset. Default is None.
    extrapolate : bool, optional
        Whether to allow extrapolation outside the table bounds. If False,
        queries outside the range raise ValueError. Default is False.

    Attributes
    ----------
    model : str
        Database version (e.g., "SLB_21").
    composition : str
        Mantle composition (e.g., "pyroliteCFMAS").
    extrapolate : bool
        Extrapolation flag.
    _tables : dict
        Dictionary mapping property names to Table objects. Keys are the
        property names from the HDF5 file (e.g., "rho", "bulk_mod",
        "shear_mod", "v_s", "v_p").

    Examples
    --------
    >>> import gdrift
    >>> # Load SLB 2021 pyrolite model
    >>> tm = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
    >>> # List available properties
    >>> print(tm.available_tables())
    ['rho', 'bulk_mod', 'shear_mod', 'v_s', 'v_p', ...]
    >>> # Forward query: temperature to Vs
    >>> vs = tm.temperature_to_vs(temperature=1600, depth=500e3)
    >>> print(f"Vs = {vs:.1f} m/s")
    >>> # Inverse query: Vs to temperature
    >>> T = tm.vs_to_temperature(vs=4500, depth=500e3)
    >>> print(f"Temperature = {T:.1f} K")
    >>> # Generic property lookup
    >>> rho = tm.temperature_to_property("rho", temperature=1600, depth=500e3)

    Notes
    -----
    - All depths are in meters from the surface
    - All temperatures are in Kelvin
    - Velocities are in m/s, densities in kg/m³, moduli in Pa
    - Not all model/composition combinations are available
    - Use `available_tables()` to see which properties are loaded
    - Vs and Vp are computed from moduli if not present in HDF5 file

    See Also
    --------
    regularise_thermodynamic_table : Smooth phase transitions
    apply_anelastic_correction : Convert to seismic-frequency velocities
    compute_swave_speed : Calculate Vs from shear modulus and density
    compute_pwave_speed : Calculate Vp from bulk/shear moduli and density

    References
    ----------
    Stixrude, L., & Lithgow-Bertelloni, C. (2011). Thermodynamics of mantle
    minerals—II. Phase equilibria. Geophysical Journal International, 184(3),
    1180-1213. https://doi.org/10.1111/j.1365-246X.2010.04890.x

    Stixrude, L., & Lithgow-Bertelloni, C. (2021). Thermal expansivity,
    heat capacity and bulk modulus of the mantle. Geophysical Journal
    International. (In preparation for SLB_21 parameters)
    """

    def __init__(self, model: str, composition: str, temps=None, depths=None, extrapolate=False):
        self.model = model
        self.composition = composition
        self.extrapolate = extrapolate

        # Load the HDF5 dataset
        loaded_model = load_dataset(dataset_name(model, composition))

        # Dictionary to store all property tables
        self._tables = {}

        # NEW GROUPED STRUCTURE: Extract from /prop group
        if 'prop' not in loaded_model:
            raise ValueError(
                f"Dataset {dataset_name(model, composition)} has invalid structure. "
                f"Expected 'prop' group not found."
            )

        prop_group = loaded_model['prop']

        # Convert pressures to depths using PREM
        pressures = prop_group['Pressures']
        depths_from_p = self._pressure_to_depth(pressures)
        temperatures = prop_group['Temperatures']

        # Store coordinate arrays for access
        self._pressures = pressures
        self._depths = depths_from_p
        self._temperatures = temperatures

        # Build tables with depth as x-coordinate
        skip_keys = {"Pressures", "Temperatures"}
        table_keys = set(prop_group.keys()) - skip_keys

        for key in table_keys:
            # In case we need to interpolate to custom grid
            if temps is not None or depths is not None:
                self._tables[key] = interpolate_table(
                    depths_from_p if depths is None else depths,
                    temperatures if temps is None else temps,
                    Table(
                        x=depths_from_p,
                        y=temperatures,
                        vals=prop_group[key],
                        name=key
                    )
                )
            else:
                self._tables[key] = Table(
                    x=depths_from_p,
                    y=temperatures,
                    vals=prop_group[key],
                    name=key
                )

    def _pressure_to_depth(self, pressures):
        """Convert pressures to depths using PREM hydrostatic equilibrium.

        Computes pressure by integrating PREM's density profile with gravity.
        """
        from .profile import PreliminaryRefEarthModel
        from .utility import compute_pressure, compute_gravity, compute_mass
        from scipy.interpolate import interp1d
        import numpy as np
        from .constants import R_earth

        prem = PreliminaryRefEarthModel()

        # Get PREM density profile
        density_profile = prem.get_profile('density')
        prem_depths = density_profile.raw_depth  # meters
        prem_densities = density_profile.raw_value  # kg/m^3

        # Convert depths to radii and reverse arrays (need center to surface)
        prem_radii = R_earth - prem_depths
        # Reverse arrays so they go from center (r=0) to surface (r=R_earth)
        prem_radii = prem_radii[::-1]
        prem_densities = prem_densities[::-1]
        prem_depths = prem_depths[::-1]

        # Compute mass and gravity at each radius
        prem_mass = compute_mass(prem_radii, prem_densities)
        prem_gravity = compute_gravity(prem_radii, prem_mass)

        # Compute pressure at each radius using hydrostatic integration
        prem_pressures = compute_pressure(prem_radii, prem_densities, prem_gravity)

        # Invert: pressure → depth
        # Sort by pressure (increasing from surface to CMB)
        sort_idx = np.argsort(prem_pressures)
        prem_pressures_sorted = prem_pressures[sort_idx]
        prem_depths_sorted = prem_depths[sort_idx]

        # Interpolate (use linear, allow extrapolation for deep mantle)
        depth_interp = interp1d(
            prem_pressures_sorted,
            prem_depths_sorted,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )

        return depth_interp(pressures)

    def get_temperatures(self):
        """Get the temperature grid for this thermodynamic model.

        Returns
        -------
        ndarray
            1D array of temperatures in Kelvin. Typically ranges from
            ~300 K to ~7000 K depending on the model.

        Notes
        -----
        Uses the temperature grid from the "shear_mod" table as a
        representative example (all tables share the same grid).
        """
        return self._tables["shear_mod"].get_y()

    def get_depths(self):
        """Get the depth grid for this thermodynamic model.

        Returns
        -------
        ndarray
            1D array of depths in meters from the surface. Typically
            ranges from 0 to 2890 km (CMB depth) for mantle models.

        Notes
        -----
        Depths are converted from the pressure grid in the HDF5 file
        using PREM hydrostatic equilibrium. Uses the depth grid from
        the "shear_mod" table as a representative example.
        """
        return self._tables["shear_mod"].get_x()

    def vs_to_temperature(self, vs: Number, depth: Number, bounds: Optional[Union[Tuple[float, float], Tuple[numpy.ndarray, numpy.ndarray]]] = (300, 7000)) -> Number:
        """
        Convert S-wave velocity (vs) to temperature at a given depth.
        Parameters:
        -----------
        vs : Number
            The S-wave velocity.
        depth : Number
            The depth at which the temperature is to be calculated.
        bounds : Optional[Union[Tuple[float, float], Tuple[numpy.ndarray, numpy.ndarray]]], default=(300, 7000)
            The bounds for the temperature calculation. It can be a tuple of floats or numpy arrays.
        Returns:
        --------
        Number
            The temperature corresponding to the given S-wave velocity and depth.
        """
        return self._v_to_temperature(vs, depth, self.compute_swave_speed(), bounds)

    def vp_to_temperature(self, vp: Number, depth: Number, bounds: Optional[Union[Tuple[float, float], Tuple[numpy.ndarray, numpy.ndarray]]] = (300, 7000)) -> Number:
        """
        Convert P-wave velocity (vp) to temperature at a given depth.
        Parameters:
        -----------
        vp : Number
            The P-wave velocity.
        depth : Number
            The depth at which the temperature is to be calculated.
        bounds : Optional[Union[Tuple[float, float], Tuple[numpy.ndarray, numpy.ndarray]]], default=(300, 7000)
            The bounds for the temperature calculation. It can be a tuple of floats or numpy arrays.
        Returns:
        --------
        Number
            The temperature corresponding to the given P-wave velocity and depth.
        """
        return self._v_to_temperature(vp, depth, self.compute_pwave_speed(), bounds)

    def available_tables(self):
        """Return a list of available table names in this model."""
        return list(self._tables.keys())

    def _get_table(self, property_name):
        """Retrieve a table by property name, computing vs/vp on the fly if needed.

        Args:
            property_name (str): Name of the property (e.g. 'rho', 'alpha', 'Cp',
                'vs' or 'v_s', 'vp' or 'v_p', or any key loaded from the HDF5 file).

        Returns:
            Table: The requested table.
        """
        if property_name in ("vs", "v_s"):
            return self.compute_swave_speed()
        elif property_name in ("vp", "v_p"):
            return self.compute_pwave_speed()
        elif property_name in self._tables:
            return self._tables[property_name]
        else:
            raise KeyError(
                f"Property '{property_name}' not found. "
                f"Available tables: {self.available_tables()}, plus 'vs'/'vp' (computed)."
            )

    def temperature_to_property(self, property_name, temperature, depth):
        """Convert temperature and depth to a material property value.

        Args:
            property_name (str): Name of the property (e.g. 'rho', 'alpha', 'Cp', 'vs', 'vp').
            temperature: Temperature value(s).
            depth: Depth value(s).

        Returns:
            Interpolated property value(s) at the given temperature and depth.
            Returns NaN if extrapolate=False and inputs are out of bounds.
        """
        table = self._get_table(property_name)

        # Get bounds
        depth_min, depth_max = table.get_x().min(), table.get_x().max()
        temp_min, temp_max = table.get_y().min(), table.get_y().max()

        # Perform interpolation
        result = LinearRectBivariateSpline(
            table.get_x(),
            table.get_y(),
            table.get_vals()).ev(depth, temperature)

        # If extrapolate is False, return NaN for out-of-bounds values
        if not self.extrapolate:
            import numpy as np
            # Convert to arrays for consistent handling
            depth_arr = np.atleast_1d(depth)
            temp_arr = np.atleast_1d(temperature)
            result_arr = np.atleast_1d(result)

            # Create mask for out-of-bounds values
            out_of_bounds = (
                (depth_arr < depth_min) | (depth_arr > depth_max) |
                (temp_arr < temp_min) | (temp_arr > temp_max)
            )

            # Set out-of-bounds values to NaN
            result_arr = np.where(out_of_bounds, np.nan, result_arr)

            # Return scalar if input was scalar
            if np.ndim(depth) == 0 and np.ndim(temperature) == 0:
                result = result_arr.item()
            else:
                result = result_arr

        return result

    def temperature_to_vs(self, temperature, depth):
        """Convert temperature and depth to shear wave velocity (Vs).

        Convenience wrapper for `temperature_to_property("vs", ...)`.
        Computes Vs from shear modulus and density using the relation:
        Vs = sqrt(shear_modulus / density).

        Parameters
        ----------
        temperature : float or array_like
            Temperature in Kelvin.
        depth : float or array_like
            Depth in meters from the surface.

        Returns
        -------
        float or ndarray
            Shear wave velocity in m/s. Returns NaN for out-of-bounds
            queries if extrapolate=False.

        Examples
        --------
        >>> import gdrift
        >>> tm = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
        >>> vs = tm.temperature_to_vs(temperature=1600, depth=500e3)
        >>> print(f"Vs = {vs:.1f} m/s")
        """
        return self.temperature_to_property("vs", temperature, depth)

    def temperature_to_vp(self, temperature, depth):
        """Convert temperature and depth to compressional wave velocity (Vp).

        Convenience wrapper for `temperature_to_property("vp", ...)`.
        Computes Vp from bulk modulus, shear modulus, and density using:
        Vp = sqrt((bulk_modulus + 4/3 * shear_modulus) / density).

        Parameters
        ----------
        temperature : float or array_like
            Temperature in Kelvin.
        depth : float or array_like
            Depth in meters from the surface.

        Returns
        -------
        float or ndarray
            Compressional wave velocity in m/s. Returns NaN for out-of-bounds
            queries if extrapolate=False.

        Examples
        --------
        >>> import gdrift
        >>> tm = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
        >>> vp = tm.temperature_to_vp(temperature=1600, depth=500e3)
        >>> print(f"Vp = {vp:.1f} m/s")
        """
        return self.temperature_to_property("vp", temperature, depth)

    def temperature_to_rho(self, temperature, depth):
        """Convert temperature and depth to density.

        Convenience wrapper for `temperature_to_property("rho", ...)`.
        Queries the pre-computed density lookup table.

        Parameters
        ----------
        temperature : float or array_like
            Temperature in Kelvin.
        depth : float or array_like
            Depth in meters from the surface.

        Returns
        -------
        float or ndarray
            Density in kg/m³. Returns NaN for out-of-bounds queries if
            extrapolate=False.

        Examples
        --------
        >>> import gdrift
        >>> tm = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
        >>> rho = tm.temperature_to_rho(temperature=1600, depth=500e3)
        >>> print(f"Density = {rho:.1f} kg/m³")
        """
        return self.temperature_to_property("rho", temperature, depth)

    def compute_swave_speed(self):
        """Compute shear wave velocity (Vs) table from elastic moduli.

        Calculates Vs at all (depth, temperature) grid points using:
        Vs = sqrt(shear_modulus / density)

        Returns
        -------
        Table
            2D table of shear wave velocities in m/s with the same
            (depth, temperature) grid as the loaded thermodynamic model.

        Notes
        -----
        This method is called automatically by `temperature_to_vs()` and
        `temperature_to_property("vs", ...)`. The computed table is cached
        internally for efficiency.

        See Also
        --------
        compute_pwave_speed : Compute Vp from bulk and shear moduli
        temperature_to_vs : Query Vs at specific (depth, temperature)
        """
        return type(self._tables["shear_mod"])(
            x=self._tables["shear_mod"].get_x(),
            y=self._tables["shear_mod"].get_y(),
            vals=compute_swave_speed(
                self._tables["shear_mod"].get_vals(),
                self._tables["rho"].get_vals(),
            ),
            name="v_s",
        )

    def compute_pwave_speed(self):
        """Compute compressional wave velocity (Vp) table from elastic moduli.

        Calculates Vp at all (depth, temperature) grid points using:
        Vp = sqrt((bulk_modulus + 4/3 * shear_modulus) / density)

        Returns
        -------
        Table
            2D table of compressional wave velocities in m/s with the same
            (depth, temperature) grid as the loaded thermodynamic model.

        Notes
        -----
        This method is called automatically by `temperature_to_vp()` and
        `temperature_to_property("vp", ...)`. The computed table is cached
        internally for efficiency.

        See Also
        --------
        compute_swave_speed : Compute Vs from shear modulus
        temperature_to_vp : Query Vp at specific (depth, temperature)
        """
        return type(self._tables["shear_mod"])(
            x=self._tables["shear_mod"].get_x(),
            y=self._tables["shear_mod"].get_y(),
            vals=compute_pwave_speed(
                self._tables["bulk_mod"].get_vals(),
                self._tables["shear_mod"].get_vals(),
                self._tables["rho"].get_vals()),
            name="v_p")

    def _v_to_temperature(self,
                          v: Number,
                          depth: Number,
                          table: Table,
                          bounds: Optional[Union[Tuple[float, float], Tuple[numpy.ndarray, numpy.ndarray]]] = (300, 7000)) -> Number:
        """
        Convert any wave speed to temperature at given depths deping on the table provided.

        Parameters:
        v (Number): wave speed.
        depth (Number): Depth at which the temperature is to be calculated.
        bounds (Optional[Union[Tuple[float, float], Tuple[numpy.ndarray, numpy.ndarray]]]):
            Bounds for the temperature search. If not provided, default bounds [300, 7000] are used.

        Returns:
        numpy.ndarray: Temperature corresponding to the given wave speed and depth.
        """

        # Convert scalar inputs to arrays for consistent processing
        is_scalar = numpy.ndim(v) == 0
        v = numpy.atleast_1d(v)
        depth = numpy.atleast_1d(depth)

        # check if bounds is a tuple of floats
        if isinstance(bounds, tuple) and all(isinstance(b, (float, int)) for b in bounds):
            bounds = tuple([numpy.full_like(v, b) for b in bounds])

        # Ensure vs, depth, and bounds are all of the same shape
        if not (v.shape == depth.shape == bounds[0].shape == bounds[1].shape):
            raise ValueError("vs, depth, and bounds must all have the same shape")

        # create a bivariate spline for the table as interpolater
        bi_spline = LinearRectBivariateSpline(
            table.get_x(),
            table.get_y(),
            table.get_vals())

        # return the temperature
        result = numpy.squeeze(
            numpy.array(
                [self._find_temperature(a_speed, a_depth, bi_spline, bounds=(lb, ub)) for a_speed, a_depth, lb, ub in zip(v, depth, bounds[0], bounds[1])]
            )
        )

        # Return scalar if input was scalar
        return result.item() if is_scalar else result

    def _find_temperature(self, val, depth, interpolator, bounds):
        def objective(temp):
            return (interpolator(depth, temp) - val)**2

        result = minimize_scalar(
            objective,
            bounds=[bounds[0], bounds[1]],
            method='bounded',
            options={'xatol': 1e-2}
        )
        return result.x if result.success else numpy.NaN


def interpolate_table(ox, oy, table_in):
    """Interpolates values from a given mineralogy table (`table_in`) to new grid points
    defined by `ox` and `oy`. The interpolation uses the nearest two neighboring points
    from the original table for each of the new grid points.

    The function normalizes the coordinates of both the input and output tables,
    constructs a KD-tree for efficient nearest-neighbor searches, and then performs
    weighted averaging based on the inverse of the distances to the nearest neighbors.

    Args:
        ox (numpy.ndarray): 1D array of x-coordinates where the output values are required.
        oy (numpy.ndarray): 1D array of y-coordinates corresponding to the x-coordinates.
        table_in (Table): An instance of a Table class, expected to have methods
            `get_x()`, `get_y()`, and `get_vals()` that return the grid coordinates and
            values of the table, respectively, and a `get_name()` method to return the
            name of the table.

    Returns:
        Table: A new instance of the Table class, containing the interpolated values
            at the grid points specified by `ox` and `oy`. This table retains the name
            of the input table.
    """
    # prepare to query for the new coordinates
    ox_x, oy_x = numpy.meshgrid(ox, oy, indexing="ij")

    ovals = LinearRectBivariateSpline(
        table_in.get_x(),
        table_in.get_y(),
        table_in.get_vals()).ev(ox_x.flatten(), oy_x.flatten())
    ovals = ovals.reshape(ox_x.shape)
    return type(table_in)(ox, oy, ovals, name=table_in.get_name())


def compute_swave_speed(shear_modulus, density):
    """ Calculate the S-wave (secondary or shear wave) speed in a material based on its
    shear modulus and density. Inputs can be floats or numpy arrays of the same size.

    Args:
        shear_modulus (float or numpy.ndarray): The shear modulus of the material,
            indicating its resistance to shear deformation.
        density (float or numpy.ndarray): The density of the material

    Returns:
        float or numpy.ndarray: The speed of S-waves in the material, calculated in meters
            per second (m/s).

    Raises:
        ValueError: If the input arguments are not all floats or not all arrays of the
            same size.
    """
    # making sure that input is either array or float
    is_either_float_or_array(shear_modulus, density)
    # This routine generates shear wave-velocities out of the loaded densy and shear modulus
    return numpy.sqrt(numpy.divide(shear_modulus, density))


def compute_pwave_speed(bulk_modulus: Number, shear_modulus: Number, density: Number) -> Number:
    """Calculate the P-wave (primary wave) speed in a material based on its bulk modulus,
    shear modulus, and density. Inputs can be floats or numpy arrays of the same size.

    Args:
        bulk_modulus (float or numpy.ndarray): The bulk modulus of the material, representing its resistance
            to uniform compression. Unit: [].
        shear_modulus (float or numpy.ndarray): The shear modulus of the material, indicating its resistance
            to shear deformation. Unit: [].
        density (float or numpy.ndarray): The density of the material, measured in kilograms per cubic meter (g/cm^3).

    Returns:
        float or numpy.ndarray: The speed of P-waves in the material, calculated in meters per second (km/s).
        If the inputs are arrays, the return will be an array of the same size.

    Notes:
    The formula used for calculating the P-wave speed is:
        Vp = sqrt((K + 4/3 * G) / rho)
    where Vp is the P-wave speed, K is the bulk modulus, G is the shear modulus,
    and rho is the density.

    """
    # making sure that input is either array or float
    is_either_float_or_array(bulk_modulus, shear_modulus, density)

    return numpy.sqrt(
        numpy.divide(
            bulk_modulus + (4. / 3.) * shear_modulus,
            density
        )
    )


def is_either_float_or_array(*args):
    if not all(isinstance(x, (float, numpy.ndarray)) for x in args):
        raise ValueError("All inputs must be either floats or numpy arrays.")

    if any(isinstance(x, numpy.ndarray) for x in args) and not all(isinstance(x, float) for x in args):
        if not all(x.shape == args[0].shape for x in args if isinstance(x, numpy.ndarray)):
            raise ValueError("All input arrays must have the same size.")


def derive_then_integrate(table: Table, temperature_profile: AbstractProfile, regular_range: Dict[str, Tuple]) -> np.ndarray:
    """
    Derives the temperature gradient, interpolates irregular values, and integrates again to obtain velocity.
    The output is anchored (= 0.) at around velocity values that are associated at temperature_profile.
    Args:
        table (object): An object containing depth and temperature data with methods `get_x()`, `get_y()`, and `get_vals()`.
        temperature_profile (object): An object with a method `at_depth(depths)` that returns temperature values at given depths.
        regular_range (dict): A dictionary with keys corresponding to table names and values as tuples indicating the acceptable range for gradients.
    Returns:
        np.ndarray: A 2D array representing the integrated velocity values adjusted for the temperature profile.
    """

    # Getting the name of the table
    key = table._name
    # Getting the depths and temperatures
    depths = table.get_x()
    temperatures = table.get_y()

    # temperature gradient
    dT = np.gradient(temperatures)

    # Creating a mesh for the depths and temperatures
    depths_x, temperatures_x = np.meshgrid(depths, temperatures, indexing="ij")

    # Getting the gradients
    dV_dT = np.gradient(table.get_vals(), depths, temperatures, axis=(0, 1))[1]

    # Finding the regular range of values (No positive jumps, no high negative jumps)
    within_range = np.logical_and(dV_dT < regular_range[key][1], dV_dT > regular_range[key][0])

    # building a tree out of the regular values
    my_tree = cKDTree(np.column_stack((depths_x[within_range].flatten(), temperatures_x[within_range].flatten())))

    # Finding the closest values to the irregular values
    distances, inds = my_tree.query(np.column_stack((depths_x[~ within_range].flatten(), temperatures_x[~ within_range].flatten())), k=3)

    # Interpolating the irregular values
    dV_dT[~within_range] = np.sum(1 / distances * dV_dT[within_range].flatten()[inds], axis=1) / np.sum(1 / distances, axis=1)

    # Integrating the derivate again to get the velocity (note that a constant needs to be found)
    V = np.cumsum(dV_dT * dT, axis=1)
    # One D profile of vs that best describes the temperature profile
    t_mean_array = np.asarray([V[i, j] for i, j in enumerate(abs(temperature_profile.at_depth(depths_x) - temperatures).argmin(axis=1))])

    # Broadcasting to the correct shape
    t_mean_array_x, _ = np.meshgrid(t_mean_array, temperatures, indexing="ij")

    # Anchoring the V-T curve at each depth for acnhor T to have zero velocity
    V -= t_mean_array_x

    return V


def regularise_thermodynamic_table(slb_pyrolite: ThermodynamicModel, temperature_profile: AbstractProfile, regular_range: Dict[str, Tuple] = default_regular_range):
    """
    Regularises the thermodynamic table by creating a regularised thermodynamic model that uses precomputed
    regular tables for S-wave and P-wave speeds.

    Args:
        slb_pyrolite (ThermodynamicModel): The original thermodynamic model.
        temperature_profile (AbstractProfile): The temperature profile to be used for regularisation. This is supposed to
            be a 1D profile of average temperature profiles.
        regular_range (Dict[str, Tuple], optional): Dictionary specifying the regularisation range for each
            parameter. Defaults to `gdrift.mineralogy.default_regular_range`.

    Returns:
        RegularisedThermodynamicModel: A regularised thermodynamic model with precomputed tables for S-wave
        and P-wave speeds.
    """
    # regular tables are a dictaionary of tables
    regular_tables = {}

    # iterating over the tables
    for table, convert_T2V in zip([slb_pyrolite._tables["rho"], slb_pyrolite.compute_swave_speed(), slb_pyrolite.compute_pwave_speed()],
                                  [slb_pyrolite.temperature_to_rho, slb_pyrolite.temperature_to_vs, slb_pyrolite.temperature_to_vp]):
        # Get name for the table
        key = table._name

        regular_tables[key] = derive_then_integrate(table, temperature_profile, regular_range)

        # the velocity for the given temperature profile
        v_average = convert_T2V(temperature=temperature_profile.at_depth(table.get_x()), depth=table.get_x())

        # Subtracting the mean
        regular_tables[key] += v_average[:, None]

    class RegularisedThermodynamicModel(ThermodynamicModel):
        """
        A wrapper class for a regularised thermodynamic model that uses precomputed regular tables
        for S-wave and P-wave speed instead of the default methods.
        """

        def __init__(self, *args, **kwargs):
            # Inherit properties from the original model
            super().__init__(*args, **kwargs)
            self._tables["rho"] = Table(self.get_depths(), self.get_temperatures(), regular_tables["rho"], name="rho")

        def compute_swave_speed(self):
            """
            Returns the regularised S-wave speed as a `Table` object.
            """
            return Table(self.get_depths(), self.get_temperatures(), regular_tables["v_s"], name="v_s")

        def compute_pwave_speed(self):
            """
            Returns the regularised P-wave speed as a `Table` object.
            """
            return Table(self.get_depths(), self.get_temperatures(), regular_tables["v_p"], name="v_p")

    return RegularisedThermodynamicModel(
        slb_pyrolite.model,
        slb_pyrolite.composition,
        slb_pyrolite.get_temperatures(),
        slb_pyrolite.get_depths())
