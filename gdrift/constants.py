"""Physical constants used throughout gdrift.

This module defines fundamental Earth structure parameters and unit conversion
factors used in geodynamic and seismic modeling calculations.

Constants
---------
R_earth : float
    Mean radius of Earth in meters (6,371,000 m). Used as the reference
    sphere radius for coordinate transformations and depth calculations.
R_cmb : float
    Radius of the core-mantle boundary in meters (3,481,000 m). Computed
    as R_earth - 2890 km, following standard seismological conventions.
celcius2kelvin : float
    Additive conversion factor from Celsius to Kelvin (273.0). Used to
    convert temperature scales in thermodynamic calculations. Note that
    most internal computations use Kelvin.

Notes
-----
These constants are consistent with the Preliminary Reference Earth Model
(PREM; Dziewonski & Anderson, 1981) and standard geophysical conventions.
For more accurate local or regional models, consider using depth-dependent
profiles from `RadialEarthModel` classes.

References
----------
Dziewonski, A. M., & Anderson, D. L. (1981). Preliminary reference Earth
model. Physics of the Earth and Planetary Interiors, 25(4), 297-356.
"""

R_earth = 6371e3  # radius of the reference sphere representing Earth
R_cmb = R_earth - 2890e3  # radius of core-mantle-boundary
celcius2kelvin = 273.0  # conversion from Celcius to Kelvin
