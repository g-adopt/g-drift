"""Utility functions for coordinate transforms, gravity, and interpolation.

This module provides mathematical and geophysical utility functions used
throughout gdrift, including:
- Coordinate system conversions (geodetic ↔ Cartesian, normalized coords)
- Gravity and pressure computation from radial density profiles
- Spatial interpolation kernels (IDW, Gaussian, Wendland, etc.)
- Fibonacci sphere for uniform sampling on spheres
- Array manipulation helpers

The coordinate transforms handle conversions between:
- **Geodetic**: (latitude, longitude, depth) in degrees and meters
- **Cartesian**: (x, y, z) in meters with origin at Earth's center
- **Normalized**: (x, y, z) scaled to R_earth for numerical stability

Key Functions
-------------
Coordinate Transforms:
  geodetic_to_cartesian : (lat, lon, depth) → (x, y, z) in meters
  cartesian_to_geodetic : (x, y, z) → (lat, lon, depth)
  nondimensionalise_coords : Scale Cartesian coords to R_earth
  dimensionalise_coords : Rescale normalized coords to meters

Geophysical Computations:
  compute_mass : Cumulative mass from radial density profile
  compute_gravity : Radial gravity from enclosed mass
  compute_pressure : Hydrostatic pressure from density and gravity

Interpolation:
  interpolate_to_points : KD-tree kernel-based interpolation
  Available kernels: "idw" (inverse distance), "gaussian", "wendland",
                     "linear", "cubic", "nearest_neighbour"

Sampling:
  fibonacci_sphere : Uniform point distribution on unit sphere

Array Helpers:
  enlist : Ensure input is numpy array
  create_labeled_array : Create named structured array
  is_ascending : Check monotonic increasing sequence
  is_descending : Check monotonic decreasing sequence

Examples
--------
>>> import gdrift
>>> import numpy as np
>>> # Coordinate transform
>>> x, y, z = gdrift.geodetic_to_cartesian(
...     lat=45.0, lon=10.0, depth=500e3)
>>> lat, lon, depth = gdrift.cartesian_to_geodetic(x, y, z)
>>>
>>> # Compute gravity profile from PREM
>>> prem = gdrift.PreliminaryRefEarthModel()
>>> rho_profile = prem.get_profile("density")
>>> depths = np.linspace(0, 2890e3, 100)
>>> radii = gdrift.constants.R_earth - depths
>>> densities = rho_profile.at_depth(depths)
>>> mass = gdrift.compute_mass(radii[::-1], densities[::-1])
>>> gravity = gdrift.compute_gravity(radii[::-1], mass)
>>>
>>> # Fibonacci sphere sampling
>>> points = gdrift.fibonacci_sphere(1000)  # 1000 points on unit sphere

Notes
-----
- All depths are measured from the surface (positive downward)
- Radii are measured from Earth's center (positive outward)
- Coordinate transforms assume spherical Earth with radius R_earth
- Gravity computation requires radius arrays starting from r=0 (center)
- Interpolation kernels have different distance decay characteristics:
  * IDW: power-law decay (customizable exponent)
  * Gaussian: exponential decay (customizable bandwidth)
  * Wendland: compact support (zero beyond cutoff radius)
  * Linear/Cubic: polynomial basis functions
  * Nearest neighbor: piecewise constant

See Also
--------
gdrift.constants : R_earth, R_cmb
gdrift.earthmodel3d : 3D interpolation using these utilities
"""

import numpy
import scipy
from .constants import R_earth, R_cmb


def is_ascending(lst):
    """Check if a list is monotonically non-decreasing.

    Parameters
    ----------
    lst : list or array_like
        Sequence to check.

    Returns
    -------
    bool
        True if lst[i] <= lst[i+1] for all i, False otherwise.
    """
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


def is_descending(lst):
    """Check if a list is monotonically non-increasing.

    Parameters
    ----------
    lst : list or array_like
        Sequence to check.

    Returns
    -------
    bool
        True if lst[i] >= lst[i+1] for all i, False otherwise.
    """
    return all(lst[i] >= lst[i + 1] for i in range(len(lst) - 1))


def compute_mass(radius, density):
    """
    Compute the mass enclosed within each radius using the cumulative trapezoidal rule.

    Args:
        radius (numpy.ndarray): Array of radii from the center of the Earth or other celestial body.
        density (numpy.ndarray): Array of densities corresponding to each radius.

    Returns:
        numpy.ndarray: Array of cumulative mass enclosed up to each radius.
    """
    if radius[0] != 0:
        raise ValueError(
            f"The first element radius should be zero, but it is {radius[0]}")

    mass_enclosed = numpy.zeros_like(radius)
    for i in range(1, len(radius)):
        shell_volume = 4 / 3 * numpy.pi * (radius[i]**3 - radius[i - 1]**3)
        average_density = (density[i] + density[i - 1]) / 2
        mass_enclosed[i] = mass_enclosed[i - 1] + shell_volume * average_density
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
    for i in range(len(radius) - 2, -1, -1):
        dr = radius[i + 1] - radius[i]
        avg_density = (density[i] + density[i + 1]) / 2
        avg_gravity = (gravity[i] + gravity[i + 1]) / 2
        pressure[i] = pressure[i + 1] + avg_density * avg_gravity * dr
    return pressure


def geodetic_to_cartesian(lat, lon, depth, earth_radius=R_earth):
    """
    Convert geographic coordinates to Cartesian coordinates.

    Parameters:
    lat (float or numpy.ndarray): Latitude in degrees.
    lon (float or numpy.ndarray): Longitude in degrees.
    depth (float or numpy.ndarray): Depth below Earth's surface in km.
    earth_radius (float): Radius of the Earth in km. Default is 6371 km.

    Returns:
    tuple: Cartesian coordinates (x, y, z).
    """
    # Convert latitude and longitude from degrees to radians
    lat_rad = numpy.radians(lat)
    lon_rad = numpy.radians(lon)

    r = earth_radius - depth
    # Compute Cartesian coordinates
    x = r * numpy.cos(lat_rad) * numpy.cos(lon_rad)
    y = r * numpy.cos(lat_rad) * numpy.sin(lon_rad)
    z = r * numpy.sin(lat_rad)

    return numpy.column_stack((x, y, z))


def cartesian_to_geodetic(x, y, z, earth_radius=R_earth):
    """
    Convert Cartesian coordinates to geographic coordinates.

    Parameters:
    x (float or numpy.ndarray): x coordinate in km.
    y (float or numpy.ndarray): y coordinate in km.
    z (float or numpy.ndarray): z coordinate in km.
    earth_radius (float): Radius of the Earth in km. Default is 6371e3 m.

    Returns:
    tuple: Geographic coordinates (lat, lon, depth).
    """
    # Compute the distance from the Earth's center
    r = numpy.sqrt(x**2 + y**2 + z**2)

    # Compute latitude in radians
    lat_rad = numpy.arcsin(z / r)

    # Compute longitude in radians
    lon_rad = numpy.arctan2(y, x)

    # Compute depth below Earth's surface
    depth = earth_radius - r

    # Convert latitude and longitude from radians to degrees
    lat = numpy.degrees(lat_rad)
    lon = numpy.degrees(lon_rad)

    return lat, lon, depth


def cartesian_to_spherical(x, y, z):
    """
    Converts Cartesian coordinates to spherical coordinates.

    Parameters:
    x (float): x-coordinate in Cartesian coordinates.
    y (float): y-coordinate in Cartesian coordinates.
    z (float): z-coordinate in Cartesian coordinates.

    Returns:
    tuple: Spherical coordinates (r, theta, phi).
    """

    # Calculate the radial distance
    r = numpy.sqrt(x**2 + y**2 + z**2)

    # Calculate the polar angle (theta)
    theta = numpy.arccos(z / r)

    # Calculate the azimuthal angle (phi)
    phi = numpy.arctan2(y, x)

    return (r, theta, phi)


def spherical_to_cartesian(r, theta, phi):
    """
    Converts spherical coordinates to Cartesian coordinates.

    Parameters:
    r (float): Radial distance in spherical coordinates.
    theta (float): Polar angle in spherical coordinates.
    phi (float): Azimuthal angle in spherical coordinates.

    Returns:
    tuple: Cartesian coordinates (x, y, z).
    """
    # Calculate the x-coordinate
    x = r * numpy.sin(theta) * numpy.cos(phi)

    # Calculate the y-coordinate
    y = r * numpy.sin(theta) * numpy.sin(phi)

    # Calculate the z-coordinate
    z = r * numpy.cos(theta)

    return (x, y, z)


def nondimensionalise_coords(x, y, z, R_nd_earth=2.22, R_nd_cmb=1.22):
    """Convert dimensional Cartesian coordinates to nondimensional form.

    Applies linear radial scaling to map physical coordinates (meters)
    to a nondimensional reference frame. Commonly used in geodynamic
    simulations to improve numerical conditioning.

    Parameters
    ----------
    x, y, z : float or array_like
        Cartesian coordinates in meters (origin at Earth's center).
    R_nd_earth : float, optional
        Nondimensional radius for Earth's surface. Default is 2.22.
    R_nd_cmb : float, optional
        Nondimensional radius for core-mantle boundary. Default is 1.22.

    Returns
    -------
    tuple of (float or ndarray)
        Nondimensionalized (x', y', z') coordinates.

    Notes
    -----
    Uses linear scaling: r' = a*r + b, where a and b are determined by
    mapping R_earth → R_nd_earth and R_cmb → R_nd_cmb. Angular coordinates
    (theta, phi) are preserved.

    See Also
    --------
    dimensionalise_coords : Inverse transformation
    """
    r, theta, phi = cartesian_to_spherical(x, y, z)

    # Calculate the slope (a)
    a = (R_nd_earth - R_nd_cmb) / (R_earth - R_cmb)
    # Calculate the intercept (b)
    b = R_nd_earth - a * R_earth

    r_scaled = a * r + b
    x_prime, y_prime, z_prime = spherical_to_cartesian(r_scaled, theta, phi)
    return (x_prime, y_prime, z_prime)


def dimensionalise_coords(x, y, z, R_nd_cmb=1.22, R_nd_earth=2.22):
    """Convert nondimensional Cartesian coordinates back to meters.

    Inverse of `nondimensionalise_coords`. Maps nondimensional coordinates
    back to physical units (meters) using linear radial scaling.

    Parameters
    ----------
    x, y, z : float or array_like
        Nondimensional Cartesian coordinates.
    R_nd_cmb : float, optional
        Nondimensional radius for core-mantle boundary. Must match the
        value used in nondimensionalisation. Default is 1.22.
    R_nd_earth : float, optional
        Nondimensional radius for Earth's surface. Must match the value
        used in nondimensionalisation. Default is 2.22.

    Returns
    -------
    tuple of (float or ndarray)
        Dimensional (x, y, z) coordinates in meters.

    Notes
    -----
    Uses inverse linear scaling: r = a*r' + b, where a and b are determined
    by mapping R_nd_earth → R_earth and R_nd_cmb → R_cmb.

    See Also
    --------
    nondimensionalise_coords : Forward transformation
    """
    r, theta, phi = cartesian_to_spherical(x, y, z)

    # Calculate the slope (a)
    a = (R_earth - R_cmb) / (R_nd_earth - R_nd_cmb)
    # Calculate the intercept (b)
    b = R_earth - a * R_nd_earth

    r_scaled = a * r + b
    x_prime, y_prime, z_prime = spherical_to_cartesian(r_scaled, theta, phi)

    return (x_prime, y_prime, z_prime)


def fibonacci_sphere(n):
    """Generates points on a sphere using the Fibonacci sphere algorithm, which
    distributes points **approximately** evenly over the surface of a sphere.

    This method calculates coordinates for each point using the golden angle,
    ensuring that each point is equidistant from its neighbors. The algorithm
    is particularly useful for creating evenly spaced points on a sphere's
    surface without clustering at the poles, a common issue in other spherical
    point distribution methods.

    Args:
        n (int): The number of points to generate on the sphere's surface.

    Returns:
        numpy.ndarray: A 2D array of shape (n, 3), where each row
                       contains the [x, y, z] coordinates of a point on the
                       sphere.

    Example:
        >>> sphere = _fibonacci_sphere(100)
        >>> print(sphere.shape)
        (100, 3)

    """

    phi = numpy.pi * (3. - numpy.sqrt(5.))  # golden angle in radians

    y = 1 - (numpy.arange(n) / (n - 1)) * 2
    radius = numpy.sqrt(1 - y * y)
    theta = phi * numpy.arange(n)
    x = numpy.cos(theta) * radius
    z = numpy.sin(theta) * radius
    return numpy.array([[x[i], y[i], z[i]] for i in range(len(x))])


def great_circle_path(lat_A, lon_A, lat_B, lon_B, n_points=360, major_arc=False):
    """Compute evenly-spaced points along a great-circle arc between two surface locations.

    Uses spherical linear interpolation (slerp) to produce a continuous
    path with no coordinate discontinuities.

    Parameters
    ----------
    lat_A, lon_A : float
        Latitude and longitude of the start point in degrees.
    lat_B, lon_B : float
        Latitude and longitude of the end point in degrees.
    n_points : int, optional
        Number of points along the arc. Default is 360.
    major_arc : bool, optional
        If True, follow the major arc (the long way around) instead of
        the minor arc. Default is False.

    Returns
    -------
    lats : numpy.ndarray
        Latitudes along the arc in degrees, shape ``(n_points,)``.
    lons : numpy.ndarray
        Longitudes along the arc in degrees, shape ``(n_points,)``.
    arc_length : float
        Total arc length in radians.

    Raises
    ------
    ValueError
        If the two points are identical (angular distance < 1e-12 rad).
    """
    lat_A_r, lon_A_r = numpy.radians(lat_A), numpy.radians(lon_A)
    lat_B_r, lon_B_r = numpy.radians(lat_B), numpy.radians(lon_B)

    a = numpy.array([
        numpy.cos(lat_A_r) * numpy.cos(lon_A_r),
        numpy.cos(lat_A_r) * numpy.sin(lon_A_r),
        numpy.sin(lat_A_r),
    ])
    b = numpy.array([
        numpy.cos(lat_B_r) * numpy.cos(lon_B_r),
        numpy.cos(lat_B_r) * numpy.sin(lon_B_r),
        numpy.sin(lat_B_r),
    ])

    dot = numpy.clip(numpy.dot(a, b), -1.0, 1.0)
    omega = numpy.arccos(dot)

    if omega < 1e-12:
        raise ValueError("Start and end points are identical.")

    sin_omega = numpy.sin(omega)

    if not major_arc:
        t = numpy.linspace(0.0, 1.0, n_points)
        coeffA = numpy.sin((1.0 - t) * omega) / sin_omega
        coeffB = numpy.sin(t * omega) / sin_omega
        pts = coeffA[:, None] * a[None, :] + coeffB[:, None] * b[None, :]
        arc_length = omega
    else:
        # Tangent direction from a toward b
        if sin_omega > 1e-12:
            e = (b - dot * a) / sin_omega
        else:
            # Antipodal: pick a canonical perpendicular
            if abs(a[2]) < 0.9:
                perp = numpy.array([0.0, 0.0, 1.0])
            else:
                perp = numpy.array([1.0, 0.0, 0.0])
            e = numpy.cross(a, perp)
            e = e / numpy.linalg.norm(e)

        # Sweep the *opposite* direction from a, arriving at b
        sweep = 2 * numpy.pi - omega
        alpha = numpy.linspace(0.0, sweep, n_points)
        pts = numpy.cos(alpha)[:, None] * a[None, :] - numpy.sin(alpha)[:, None] * e[None, :]
        arc_length = sweep

    lats = numpy.degrees(numpy.arcsin(numpy.clip(pts[:, 2], -1.0, 1.0)))
    lons = numpy.degrees(numpy.arctan2(pts[:, 1], pts[:, 0]))
    return lats, lons, arc_length


def great_circle_cross_section(lat_A, lon_A, lat_B, lon_B,
                               n_arc=360, n_depth=60,
                               major_arc=False,
                               min_depth=0.0, max_depth=None):
    """Build a 2-D cross-section grid along a great-circle arc.

    The returned grids are ready for polar-projection plotting (theta vs r)
    and the query coordinates can be passed directly to
    :meth:`~gdrift.EarthModel3D.at`.

    Parameters
    ----------
    lat_A, lon_A : float
        Latitude and longitude of the start point (degrees).
    lat_B, lon_B : float
        Latitude and longitude of the end point (degrees).
    n_arc : int, optional
        Number of points along the arc. Default is 360.
    n_depth : int, optional
        Number of depth levels. Default is 60.
    major_arc : bool, optional
        If True, follow the major arc. Default is False.
    min_depth : float, optional
        Minimum depth in meters. Default is 0.
    max_depth : float, optional
        Maximum depth in meters. Default is ``R_earth - R_cmb`` (full mantle).

    Returns
    -------
    theta_grid : numpy.ndarray
        Angular distance from the arc midpoint in radians, shape
        ``(n_depth, n_arc)``.  Negative values are toward point A,
        positive toward point B.
    r_grid : numpy.ndarray
        Radial distance from Earth's centre in meters, shape
        ``(n_depth, n_arc)``.
    query_coords : numpy.ndarray
        Cartesian coordinates of shape ``(n_depth * n_arc, 3)``
        suitable for :meth:`~gdrift.EarthModel3D.at`.
    arc_info : dict
        Metadata with keys ``arc_length`` (radians), ``major_arc`` (bool),
        ``midpoint_lat``, ``midpoint_lon``, ``endpoint_A`` and
        ``endpoint_B`` (tuples).
    """
    if max_depth is None:
        max_depth = R_earth - R_cmb

    lats, lons, arc_length = great_circle_path(
        lat_A, lon_A, lat_B, lon_B, n_points=n_arc, major_arc=major_arc
    )

    mid_idx = n_arc // 2
    midpoint_lat = float(lats[mid_idx])
    midpoint_lon = float(lons[mid_idx])

    theta_1d = numpy.linspace(-arc_length / 2.0, arc_length / 2.0, n_arc)
    depths_1d = numpy.linspace(min_depth, max_depth, n_depth)
    theta_grid, depth_grid = numpy.meshgrid(theta_1d, depths_1d)
    r_grid = R_earth - depth_grid

    # Tile surface lats/lons across all depth levels
    lats_2d = numpy.tile(lats, (n_depth, 1))
    lons_2d = numpy.tile(lons, (n_depth, 1))

    query_coords = geodetic_to_cartesian(
        lats_2d.ravel(), lons_2d.ravel(), depth_grid.ravel()
    )

    arc_info = {
        "arc_length": arc_length,
        "major_arc": major_arc,
        "midpoint_lat": midpoint_lat,
        "midpoint_lon": midpoint_lon,
        "endpoint_A": (lat_A, lon_A),
        "endpoint_B": (lat_B, lon_B),
    }

    return theta_grid, r_grid, query_coords, arc_info


def enlist(obj):
    """ Enlist makes sure we have a list

    Args:
        obj of any kind

    Returns:
        a list
    """
    if isinstance(obj, list):
        return obj
    else:
        return [obj]


def interpolate_to_points(values, distances, inds, min_distance=1e-6):
    """
    Interpolate field data to given query points using weighted averaging.

    Parameters
    ----------
    values : np.ndarray
        Array of shape (n, ...) containing the field data to interpolate from.
    distances : np.ndarray
        Array of shape (n, k) containing the distances to the k nearest neighbors.
    inds : np.ndarray
        Array of shape (n, k) containing the indices of the k nearest neighbors.
    min_distance : float, optional
        Minimum distance to avoid division by zero. Default is 1e-6.

    Returns
    -------
    np.ndarray
        Interpolated field data at the query points.
    """
    safe_dists = numpy.where(distances < min_distance, min_distance, distances)
    replace_flg = distances[:, 0] < min_distance

    with numpy.errstate(divide='ignore', invalid='ignore'):
        if len(values.shape) > 1:
            weights = 1 / safe_dists
            weighted_sum = numpy.einsum("ij, ijk -> ik", weights, values[inds])
            ret = weighted_sum / numpy.sum(weights, axis=1)[:, numpy.newaxis]
            ret[replace_flg, :] = values[inds[replace_flg, 0], :]
        else:
            weights = 1 / safe_dists
            weighted_sum = numpy.einsum("ij, ij -> i", weights, values[inds])
            ret = weighted_sum / numpy.sum(weights, axis=1)
            ret[replace_flg] = values[inds[replace_flg, 0]]

    return ret


def create_labeled_array(data_dict, labels):
    """
    Create a labeled array from a dictionary of arrays and a list of labels.

    Args:
        data_dict (dict): Dictionary where keys are labels and values are arrays of length n.
        labels (list): List of strings representing the labels.

    Returns:
        numpy.ndarray: Array of shape (n, m) where m is the number of labels.
    """
    n = len(next(iter(data_dict.values())))
    m = len(labels)
    labeled_array = numpy.zeros((n, m))

    for i, label in enumerate(labels):
        if label in data_dict:
            labeled_array[:, i] = data_dict[label]
        else:
            raise ValueError(f"Label '{label}' not found in data dictionary.")

    return labeled_array


def create_data_dict(labeled_array, labels):
    """
    Create a dictionary of arrays from a labeled array and a list of labels.

    Args:
        labeled_array (numpy.ndarray): Array of shape (n, m) where m is the number of labels.
        labels (list): List of strings representing the labels.

    Returns:
        dict: Dictionary where keys are labels and values are arrays of length n.
    """
    if labeled_array.shape[1] != len(labels):
        raise ValueError("Number of columns in labeled_array must match the number of labels.")

    data_dict = {}
    for i, label in enumerate(labels):
        data_dict[label] = labeled_array[:, i]

    return data_dict
