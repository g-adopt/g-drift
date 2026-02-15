"""Geodynamic adiabatic temperature profile computation.

This module provides functions for computing 1D adiabatic temperature profiles
through the mantle using self-consistent thermodynamic models. The adiabatic
gradient is integrated from a surface potential temperature down to the
core-mantle boundary (CMB), yielding temperature and material property profiles
that represent the reference state for mantle convection studies.

Key Functions
-------------
prem_gravity_profile : Build a gravity-vs-depth spline from PREM
compute_adiabat : Integrate the adiabatic gradient ODE and evaluate properties

Examples
--------
>>> import gdrift
>>> gravity = gdrift.prem_gravity_profile()
>>> print(f"Surface gravity: {gravity.at_depth(0):.2f} m/s^2")
>>>
>>> tm = gdrift.ThermodynamicModel("SLB_21", "pyroliteCFMAS")
>>> result = gdrift.compute_adiabat(tm, T0=1600)
>>> print(f"CMB temperature: {result['temperature'][-1]:.0f} K")

References
----------
Stixrude, L., & Lithgow-Bertelloni, C. (2021). Thermal expansivity, heat
capacity and bulk modulus of the mantle. Geophysical Journal International.
"""

import numpy as np
import scipy.integrate

from .constants import R_earth, R_cmb
from .profile import PreliminaryRefEarthModel, SplineProfile
from .utility import compute_mass, compute_gravity


def prem_gravity_profile(n_points=1000):
    """Build a gravity-versus-depth spline from PREM.

    Loads the Preliminary Reference Earth Model (PREM), computes enclosed
    mass and gravitational acceleration as a function of radius, and returns
    a SplineProfile of gravity versus depth.

    Parameters
    ----------
    n_points : int, optional
        Number of radial points for the integration grid. Default is 1000.

    Returns
    -------
    SplineProfile
        Gravity profile (m/s^2) as a function of depth (m) from the surface.
        Extrapolation is enabled for robustness near boundaries.
    """
    prem = PreliminaryRefEarthModel()
    radii = np.linspace(0, R_earth, n_points)
    depths = R_earth - radii
    mass = compute_mass(radius=radii, density=prem.at_depth("density", depths))
    gravity = compute_gravity(radii, mass)
    return SplineProfile(depth=depths, value=gravity, name="gravity", extrapolate=True)


def compute_adiabat(thermo_model, T0, depths=None, gravity_profile=None,
                    property_names=None, rtol=1e-3):
    """Compute a 1D adiabatic temperature profile through the mantle.

    Integrates the adiabatic gradient ODE:

        dT/dz = alpha * T * g / Cp_SI

    where Cp_SI = Cp / (rho * V) is the specific heat capacity in SI units
    (J kg^-1 K^-1), alpha is thermal expansivity, T is temperature, and g is
    gravitational acceleration.

    After integration, material properties are evaluated along the adiabat and
    derived quantities (SI heat capacities, dissipation number) are computed.

    Parameters
    ----------
    thermo_model : ThermodynamicModel
        Thermodynamic lookup table (e.g., SLB_21 pyroliteCFMAS).
    T0 : float
        Surface potential temperature in Kelvin.
    depths : array_like, optional
        Depth grid in meters. Default is ``np.linspace(0, R_earth - R_cmb, 257)``.
    gravity_profile : SplineProfile, optional
        Gravity-vs-depth profile. If None, computed from PREM via
        ``prem_gravity_profile()``.
    property_names : list of str, optional
        Thermodynamic properties to evaluate along the adiabat. Default is
        ``["rho", "alpha", "Cp", "V", "Cv", "beta", "gamma"]``.
    rtol : float, optional
        Relative tolerance for the ODE integrator. Default is 1e-3.

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``"depths"`` : ndarray — depth grid (m)
        - ``"temperature"`` : ndarray — adiabatic temperature (K)
        - ``"rho"`` : ndarray — density (kg/m^3)
        - ``"alpha"`` : ndarray — thermal expansivity (1/K)
        - ``"Cp"`` : ndarray — molar heat capacity at constant pressure
        - ``"V"`` : ndarray — molar volume
        - ``"Cv"`` : ndarray — molar heat capacity at constant volume
        - ``"beta"`` : ndarray — isothermal compressibility
        - ``"gamma"`` : ndarray — Grueneisen parameter
        - ``"Cp_SI"`` : ndarray — specific heat capacity Cp (J/kg/K)
        - ``"Cv_SI"`` : ndarray — specific heat capacity Cv (J/kg/K)
        - ``"gravity"`` : ndarray — gravitational acceleration (m/s^2)
        - ``"Di"`` : float — dissipation number
    """
    if depths is None:
        depths = np.linspace(0, R_earth - R_cmb, 257)
    depths = np.asarray(depths)

    if gravity_profile is None:
        gravity_profile = prem_gravity_profile()

    if property_names is None:
        property_names = ["rho", "alpha", "Cp", "V", "Cv", "beta", "gamma"]

    # Get table bounds for clamping — the adaptive ODE integrator may
    # evaluate dT/dz at trial points beyond the requested depth grid.
    table_depths = thermo_model.get_depths()
    table_temps = thermo_model.get_temperatures()
    depth_min, depth_max = float(table_depths[0]), float(table_depths[-1])
    temp_min, temp_max = float(table_temps[0]), float(table_temps[-1])

    def dT_dz(T, depth):
        # Clamp to table bounds so trial evaluations don't return NaN
        depth_c = np.clip(depth, depth_min, depth_max)
        T_c = np.clip(T, temp_min, temp_max)
        alpha = thermo_model.temperature_to_property("alpha", T_c, depth_c)
        rho = thermo_model.temperature_to_property("rho", T_c, depth_c)
        V = thermo_model.temperature_to_property("V", T_c, depth_c)
        Cp = thermo_model.temperature_to_property("Cp", T_c, depth_c)
        Cp_SI = Cp / (rho * V)
        return alpha * T_c * gravity_profile.at_depth(depth_c) / Cp_SI

    temperature = scipy.integrate.odeint(
        func=dT_dz, y0=T0, t=depths, rtol=rtol,
    ).squeeze()

    result = {
        "depths": depths,
        "temperature": temperature,
    }

    for name in property_names:
        result[name] = thermo_model.temperature_to_property(
            name, temperature, depths,
        )

    # Derived SI heat capacities
    if "Cp" in result and "rho" in result and "V" in result:
        result["Cp_SI"] = result["Cp"] / (result["rho"] * result["V"])
    if "Cv" in result and "rho" in result and "V" in result:
        result["Cv_SI"] = result["Cv"] / (result["rho"] * result["V"])

    # Gravity along the profile
    result["gravity"] = gravity_profile.at_depth(depths)

    # Dissipation number: Di = alpha_s * g_s * D / Cp_SI_s
    D = R_earth - R_cmb
    if "alpha" in result and "Cp_SI" in result:
        result["Di"] = (result["alpha"][0] * result["gravity"][0] * D
                        / result["Cp_SI"][0])

    return result
