import numpy
import scipy
import math
from .constants import R_earth, R_cmb

def is_ascending(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

def is_descending(lst):
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


def geodetic_to_cartesian(lat, lon, r):
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

    # Compute Cartesian coordinates
    x = r * numpy.cos(lat_rad) * numpy.cos(lon_rad)
    y = r * numpy.cos(lat_rad) * numpy.sin(lon_rad)
    z = r * numpy.sin(lat_rad)

    return x, y, z


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
    r, theta, phi = cartesian_to_spherical(x, y, z)

    # Calculate the slope (a)
    a = (R_nd_earth - R_nd_cmb) / (R_earth - R_cmb)
    # Calculate the intercept (b)
    b = R_nd_earth - a * R_earth

    r_scaled = a * r + b
    x_prime, y_prime, z_prime = spherical_to_cartesian(r_scaled, theta, phi)
    return (x_prime, y_prime, z_prime)


def dimensionalise_coords(x, y, z, R_nd_cmb=1.22, R_nd_earth=2.22):
    """
    """
    r, theta, phi = cartesian_to_spherical(x, y, z)

    # Calculate the slope (a)
    a = (R_earth - R_cmb)/ (R_nd_earth - R_nd_cmb)
    # Calculate the intercept (b)
    b = R_earth - a * R_nd_earth

    r_scaled = a * r + b
    x_prime, y_prime, z_prime = spherical_to_cartesian(r_scaled, theta, phi)

    return (x_prime, y_prime, z_prime)
