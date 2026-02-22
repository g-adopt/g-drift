"""Anelastic corrections for converting between elastic and anelastic seismic velocities.

This module implements frequency-dependent anelastic attenuation corrections
that account for the difference between seismic wave velocities measured at
seismic frequencies (~1 Hz) and elastic velocities computed from mineral
physics at high frequencies (effective infinite frequency limit).

Anelasticity causes seismic waves to attenuate and disperse as they propagate
through the mantle. The quality factor Q quantifies this attenuation, with
lower Q indicating stronger anelastic effects. Velocity corrections depend on
Q, temperature, pressure, and the homologous temperature (T/T_solidus).

Two parameterizations are provided:
1. **Cammarano et al. (2003)**: Uses parameters B (grain size), g (activation
   volume), and a (frequency exponent) to compute Q(depth, temperature)
2. **Goes et al. (2000)**: Uses activation energy H* and activation volume V*
   with pressure from PREM to compute Q(depth, temperature) for the upper mantle

Key Classes
-----------
BaseAnelasticityModel : Abstract base class for anelastic models
CammaranoAnelasticityModel : B, g, a parameterization (6 Q-profiles: Q1-Q6)
GoesAnelasticityModel : Activation energy parameterization (2 Q-profiles: Q1, Q2)

Key Functions
-------------
apply_anelastic_correction : Apply anelastic correction to ThermodynamicModel
BaseAnelasticityModel.build_ghelichkhan_solidus : Factory for Hirschmann solidus

Examples
--------
>>> import gdrift
>>> # Load elastic thermodynamic model
>>> tm = gdrift.ThermodynamicModel("SLB_21_pyroliteCFMAS")
>>>
>>> # Create Cammarano Q3 anelasticity model
>>> anelastic = gdrift.CammaranoAnelasticityModel.from_q_profile("Q3")
>>>
>>> # Apply correction to get anelastic (seismic-frequency) model
>>> tm_anelastic = gdrift.apply_anelastic_correction(tm, anelastic)
>>>
>>> # Now tm_anelastic.temperature_to_vs returns seismic-frequency Vs
>>> vs_elastic = tm.temperature_to_vs(1600, 500e3)
>>> vs_anelastic = tm_anelastic.temperature_to_vs(1600, 500e3)
>>> print(f"Elastic: {vs_elastic:.1f} m/s, Anelastic: {vs_anelastic:.1f} m/s")
>>>
>>> # Use Goes Q1 model instead
>>> anelastic_goes = gdrift.GoesAnelasticityModel.from_q_profile("Q1")
>>> tm_anelastic_q1 = gdrift.apply_anelastic_correction(tm, anelastic_goes)

Notes
-----
- Anelastic corrections typically reduce Vs by 1-3% in the mantle
- Corrections are largest at high temperatures (near solidus)
- Q-profiles (Q1-Q6) represent different assumptions about grain size,
  water content, and attenuation mechanisms
- Cammarano Q3 and Goes Q1 are commonly used in geodynamic studies
- The correction assumes a reference frequency of 1 Hz for seismic waves

References
----------
Cammarano, F., Goes, S., Vacher, P., & Giardini, D. (2003). Inferring
upper-mantle temperatures from seismic velocities. Physics of the Earth
and Planetary Interiors, 138(3-4), 197-222.

Goes, S., Govers, R., & Vacher, P. (2000). Shallow mantle temperatures
under Europe from P and S wave tomography. Journal of Geophysical Research,
105(B5), 11153-11169.

See Also
--------
gdrift.mineralogy.ThermodynamicModel : Elastic velocity lookup tables
gdrift.profile.HirschmannSolidus : Solidus temperature profile
"""

from abc import ABC, abstractmethod
import warnings
import numpy
import numpy.typing as npt
import scipy.interpolate
from typing import TypeVar, Callable

from .profile import SplineProfile, RadialEarthModelFromFile, HirschmannSolidus, PreliminaryRefEarthModel
from .utility import compute_mass, compute_gravity, compute_pressure
from .constants import R_earth

AnelasticityModel = TypeVar("AnelasticityModel", bound="BaseAnelasticityModel")


class BaseAnelasticityModel(ABC):
    """
    Abstract base class for an anelasticity model.
    All anelasticity models must be able to compute a Q matrix given depths and temperatures.
    """

    @abstractmethod
    def compute_Q_shear(self, depths: npt.ArrayLike, temperatures: npt.ArrayLike) -> npt.NDArray:
        """
        Computes the s-wave anelastic quality factor (Q) matrix for given depths and temperatures.

        Args:
            depths (numpy.ndarray): Array of depths at which Q values are required.
            temperatures (numpy.ndarray): Array of temperatures corresponding to the depths.

        Returns:
            numpy.ndarray: A matrix of Q values corresponding to the given depths and temperatures.
        """
        pass

    @abstractmethod
    def compute_Q_bulk(self, depths: npt.ArrayLike, temperatures: npt.ArrayLike) -> npt.NDArray:
        """
        Computes the compressional anelastic quality factor (Q) matrix for given depths and temperatures.

        Args:
            depths (numpy.ndarray): Array of depths at which Q values are required.
            temperatures (numpy.ndarray): Array of temperatures corresponding to the depths.

        Returns:
            numpy.ndarray: A matrix of Q values corresponding to the given depths and temperatures.
        """
        pass

    @staticmethod
    def build_ghelichkhan_solidus() -> SplineProfile:
        """Construct the composite Ghelichkhan et al. (2021) solidus by combining
        Hirschmann (shallow) + Andrault (deep) profiles.

        Returns:
            SplineProfile: The composite solidus profile, extrapolated and extended to 3000 km.
        """
        andrault_solidus = RadialEarthModelFromFile(
            model_name="1d_solidus_Andrault_et_al_2011_EPSL",
            description="Andrault et al. 2011, EPSL"
        )
        hirsch_solidus = HirschmannSolidus()

        my_depths = []
        my_solidus = []
        for solidus_model in [
            hirsch_solidus.get_profile("solidus temperature"),
            andrault_solidus.get_profile("solidus temperature")
        ]:
            d_min, d_max = solidus_model.min_max_depth()
            dpths = numpy.arange(d_min, d_max, 10e3)
            my_depths.extend(dpths)
            my_solidus.extend(solidus_model.at_depth(dpths))

        # Extend to 3000 km to avoid extrapolation issues
        my_depths.extend([3000e3])
        my_solidus.extend([solidus_model.at_depth(dpths[-1])])

        return SplineProfile(
            depth=numpy.asarray(my_depths),
            value=numpy.asarray(my_solidus),
            extrapolate=True,
            name="Ghelichkhan et al. 2021"
        )


class CammaranoAnelasticityModel(BaseAnelasticityModel):
    """
    A specific implementation of an anelasticity model following the approach by Cammarano et al.
    """

    def __init__(
        self,
        B: Callable,
        g: Callable,
        a: Callable,
        solidus: SplineProfile,
        Q_bulk: Callable = lambda x: 10000,
        omega: Callable = lambda x: 1.0,
    ):
        """
        Initialize the model with the given parameters.

        Args:
            B (Callable): Scaling factor for the Q model.
            g (Callable): Activation energy parameter.
            a (Callable): Frequency dependency parameter.
            solidus (SplineProfile): Solidus temperature profile for mantle.
            Q_bulk (Callable): Bulk quality factor (default is 10000).
            omega (Callable): Seismic frequency (default is 1).
        """
        self.B = B
        self.g = g
        self.a = a
        self.omega = omega
        self.solidus = solidus
        self.Q_bulk = Q_bulk

    @classmethod
    def from_q_profile(cls, q_profile: str) -> "CammaranoAnelasticityModel":
        """Create a CammaranoAnelasticityModel from a predefined Q-profile.

        Uses the six Q-profiles (Q1-Q6) from Cammarano et al. (2003), with
        upper/lower mantle parameter switching at 660 km depth. The Q factor
        is computed as:

            Q = B * omega^a * exp(a * g * T_solidus / T)

        where B controls the overall attenuation amplitude (related to grain
        size), g is a dimensionless activation energy parameter, a is the
        frequency exponent, and T_solidus is the solidus temperature at the
        given depth.

        Q-profiles represent different assumptions about grain size, water
        content, and attenuation mechanisms in the mantle:

        ======= ============= ============= ======================================
        Profile B (UM / LM)   g (UM / LM)   Physical Interpretation
        ======= ============= ============= ======================================
        Q1      0.5 / 10      20 / 10       Fine grain size, low water content
        Q2      0.8 / 15      20 / 10       Medium grain size
        Q3      1.1 / 20      20 / 10       Coarse grain, higher water content
        Q4      0.035 / 2.25  30 / 15       SLB activation energy, fine grain
        Q5      0.056 / 3.6   30 / 15       SLB activation energy, medium grain
        Q6      0.077 / 4.95  30 / 15       SLB activation energy, coarse grain
        ======= ============= ============= ======================================

        UM = upper mantle (< 660 km), LM = lower mantle (>= 660 km).
        Q1-Q3 use Cammarano et al. (2003) activation energy parameterization.
        Q4-Q6 use Stixrude & Lithgow-Bertelloni activation energy values.
        All profiles use a = 0.2 (frequency exponent) and omega = 1.0 Hz.

        Args:
            q_profile (str): One of "Q1" through "Q6".

        Returns:
            CammaranoAnelasticityModel: The configured model.

        Raises:
            ValueError: If q_profile is not one of Q1-Q6.
        """
        parameters = {
            "Q1": {"B": [0.5, 10], "g": [20, 10]},
            "Q2": {"B": [0.8, 15], "g": [20, 10]},
            "Q3": {"B": [1.1, 20], "g": [20, 10]},
            "Q4": {"B": [0.035, 2.25], "g": [30, 15]},
            "Q5": {"B": [0.056, 3.6], "g": [30, 15]},
            "Q6": {"B": [0.077, 4.95], "g": [30, 15]},
        }
        if q_profile not in parameters:
            raise ValueError(f"Unknown Q-profile '{q_profile}'. Choose from {list(parameters.keys())}.")

        p = parameters[q_profile]
        solidus = BaseAnelasticityModel.build_ghelichkhan_solidus()

        def B(x):
            return numpy.where(x < 660e3, p["B"][0], p["B"][1])

        def g(x):
            return numpy.where(x < 660e3, p["g"][0], p["g"][1])

        def a(x):
            return 0.2

        def omega(x):
            return 1.0

        def Q_kappa(x):
            return numpy.where(x < 660e3, 1e3, 1e4)

        return cls(B=B, g=g, a=a, solidus=solidus, Q_bulk=Q_kappa, omega=omega)

    def compute_Q_shear(self, depths: npt.ArrayLike, temperatures: npt.ArrayLike) -> npt.NDArray:
        """
        Compute the shear Q (attenuation quality factor) matrix based on input depths and temperatures.

        Args:
            depths (numpy.ndarray): An array of depths at which Q values are to be calculated.
            temperatures (numpy.ndarray): An array of temperatures corresponding to the specified depths.

        Returns:
            numpy.ndarray: A matrix of calculated Q values, representing the shear attenuation quality factor.
        """
        depths = numpy.asarray(depths)
        temperatures = numpy.asarray(temperatures)

        Q_values = (
            self.B(depths) * (self.omega(depths)**self.a(depths)) * numpy.exp(
                (self.a(depths) * self.g(depths) * self.solidus.at_depth(depths)) / temperatures)
        )

        return Q_values

    def compute_Q_bulk(self, depths: npt.ArrayLike, temperatures: npt.ArrayLike) -> npt.NDArray:
        """
        Compute the bulk Q factor.

        Args:
            depths: Array of depths.
            temperatures: Array of temperatures (unused for bulk Q in Cammarano model).
        """
        return self.Q_bulk(depths)


class GoesAnelasticityModel(BaseAnelasticityModel):
    """Anelasticity model following Goes et al. (2000, JGR).

    Uses activation energy and activation volume to compute the shear quality
    factor Q_mu via:

        Q_mu = A * omega^a * exp(a * (H* + P * V*) / (R * T))

    where A is a pre-exponential factor, H* is activation energy (J/mol),
    V* is activation volume (m^3/mol), P is pressure from PREM, R is the
    gas constant, and T is temperature (K).

    This model was calibrated for the upper mantle (50-200 km depth). Below
    660 km depth, Q is set to a very large value (effectively no attenuation).
    """

    _nd_radial = 1000  # number of radial points for PREM pressure interpolant

    def __init__(self, A, H_star, V_star, a, omega=1.0, Q_bulk=1000.0, max_depth=660e3):
        """
        Initialize the Goes anelasticity model.

        Args:
            A (float): Pre-exponential scaling factor.
            H_star (float): Activation energy in J/mol.
            V_star (float): Activation volume in m^3/mol.
            a (float): Frequency exponent.
            omega (float): Seismic frequency in Hz (default 1.0).
            Q_bulk (float): Constant bulk Q (default 1000, Durek & Ekstrom 1996).
            max_depth (float): Depth below which Q is set very high (default 660e3 m).
        """
        self.A = A
        self.H_star = H_star
        self.V_star = V_star
        self._a_value = a
        self.a = lambda x: a  # callable for compatibility with apply_anelastic_correction
        self.omega = omega
        self.Q_bulk = Q_bulk
        self.max_depth = max_depth
        self._warned_deep = False
        self._setup_depth_to_pressure()

    def _setup_depth_to_pressure(self):
        """Build a PREM-based depth-to-pressure interpolant."""
        prem = PreliminaryRefEarthModel()
        radius = numpy.linspace(0.0, R_earth, self._nd_radial)
        depths = R_earth - radius
        mass = compute_mass(radius, prem.at_depth("density", depths))
        gravity = compute_gravity(radius, mass)
        pressure = compute_pressure(radius, prem.at_depth("density", depths), gravity)
        self._depth_to_pressure = scipy.interpolate.interp1d(depths, pressure, kind="linear")

    @classmethod
    def from_q_profile(cls, q_profile: str) -> "GoesAnelasticityModel":
        """Create a GoesAnelasticityModel from a predefined Q-profile.

        Uses the Goes et al. (2000) parameterization from Table A2 where the
        Q factor is computed as:

            Q_mu = A * omega^a * exp(a * (H* + P * V*) / (R * T))

        ======= ====== ========== ============= ===============
        Profile a      A          H* (kJ/mol)   V* (cm^3/mol)
        ======= ====== ========== ============= ===============
        Q1      0.15   0.148      500           20
        Q2      0.25   2.0e-4     584           21
        ======= ====== ========== ============= ===============

        Args:
            q_profile (str): One of "Q1" or "Q2".

        Returns:
            GoesAnelasticityModel: The configured model.

        Raises:
            ValueError: If q_profile is not Q1 or Q2.
        """
        parameters = {
            "Q1": {"A": 0.148, "H_star": 500e3, "V_star": 20e-6, "a": 0.15},
            "Q2": {"A": 2.0e-4, "H_star": 584e3, "V_star": 21e-6, "a": 0.25},
        }
        if q_profile not in parameters:
            raise ValueError(f"Unknown Q-profile '{q_profile}'. Choose from {list(parameters.keys())}.")

        return cls(**parameters[q_profile])

    def compute_Q_shear(self, depths: npt.ArrayLike, temperatures: npt.ArrayLike) -> npt.NDArray:
        """Compute the shear Q using the Goes et al. (2000) activation energy formulation.

        Args:
            depths (numpy.ndarray): Array of depths in meters.
            temperatures (numpy.ndarray): Array of temperatures in Kelvin.

        Returns:
            numpy.ndarray: Shear quality factor Q matrix.
        """
        depths = numpy.asarray(depths, dtype=float)
        temperatures = numpy.asarray(temperatures, dtype=float)

        if not self._warned_deep and numpy.any(depths > self.max_depth):
            warnings.warn(
                "Goes et al. (2000) was calibrated for the upper mantle only "
                f"(depths <= {self.max_depth/1e3:.0f} km). Depths beyond this "
                "are assigned Q = 1e10 (no attenuation).",
                stacklevel=2,
            )
            self._warned_deep = True

        pressure = self._depth_to_pressure(numpy.clip(depths, 0, R_earth))
        R_gas = 8.314  # J/(mol·K)
        Q = self.A * (self.omega ** self._a_value) * numpy.exp(
            self._a_value * (self.H_star + pressure * self.V_star) / (R_gas * temperatures)
        )
        Q = numpy.where(depths > self.max_depth, 1e10, Q)
        return Q

    def compute_Q_bulk(self, depths: npt.ArrayLike, temperatures: npt.ArrayLike) -> npt.NDArray:
        """Compute the bulk Q factor (constant).

        Args:
            depths: Array of depths (unused).
            temperatures: Array of temperatures (unused).

        Returns:
            float: Constant bulk Q value.
        """
        return self.Q_bulk


def apply_anelastic_correction(
    thermo_model,
    anelastic_model: BaseAnelasticityModel,
):
    r"""
    Apply anelastic corrections to seismic velocity data using the provided "anelastic_model"
    within the low attenuation limit. The corrections are based on the equation:
        $1 - \frac{V(anelastic)}{V(elastic)} = \frac{1}{2} \cot(\frac{\alpha \pi}{2}) Q^{-1}$
    as described by Stixrude & Lithgow-Bertelloni (doi:10.1029/2004JB002965, Eq-10).

        thermo_model (ThermodynamicModel): The thermodynamic model containing temperature and depth data.

        ThermodynamicModel: A new thermodynamic model with anelastically corrected seismic velocities.

    The returned model includes the following methods with anelastic corrections:

    - compute_swave_speed: Calculates the anelastic effect on shear wave speed.
    - compute_pwave_speed: Calculates the anelastic effect on compressional wave speed.

    The `compute_swave_speed` method applies the anelastic correction to the shear wave speed using the provided
    anelastic model. The `compute_pwave_speed` method applies the anelastic correction to the compressional wave speed
    using the quality factor derived from the equations provided by Don L. Anderson & R. S. Hart (1978, PEPI eq 1-3).

    The corrections are applied by meshing the depths and temperatures from the thermodynamic model and computing the
    quality factor matrices for shear and bulk moduli. The corrected seismic velocities are then calculated and returned
    as new tables with the corrected values.
    """
    class ThermodynamicModelPrime(thermo_model.__class__):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute_swave_speed(self):
            """
            Computes the anelastically corrected shear wave speed.
            This method first retrieves the shear wave speed table from the superclass.
            It then creates a meshgrid of depths and temperatures based on the table's
            x and y values. Using these grids, it computes the shear wave quality factor
            (Q) matrix using the provided anelastic model. The shear wave speed values
            are then corrected for anelasticity using the computed Q matrix and the
            anelastic model's parameter 'a'. The corrected shear wave speed values are
            returned in a new table with the same x and y values but updated values and
            name.

            Returns:
                A table of anelastically corrected shear wave speed values.
            """
            #
            swave_speed_table = super().compute_swave_speed()
            depths_x, temperatures_x = numpy.meshgrid(
                swave_speed_table.get_x(),
                swave_speed_table.get_y(),
                indexing="ij")

            # For shear wave velocity Q = Q_\mu
            Q_matrix = anelastic_model.compute_Q_shear(
                depths_x, temperatures_x)

            # Anelastically corrected shear wave values
            corrected_vals = (
                swave_speed_table.get_vals() * (1 - 0.5 / numpy.tan(anelastic_model.a(depths_x) * numpy.pi / 2) / Q_matrix)
            )
            #
            return type(swave_speed_table)(
                x=swave_speed_table.get_x(),
                y=swave_speed_table.get_y(),
                vals=corrected_vals,
                name=f"{swave_speed_table.get_name()}"
            )

        def compute_pwave_speed(self):
            """
            Calculate the anelastic effect on compressional wave speed.

            This method replaces the original `compute_pwave_speed` function by incorporating
            the anelastic effect on compressional wave speed. The quality factor for the
            P - wave speed is derived from the equations provided by Don L. Anderson & R. S. Hart
            (1978, PEPI eq 1 - 3):

                Q_s = Q_{\\mu}
                \frac{1}{Q_p} = \frac{L}{Q_\\mu} + \frac{(1-L)}{Q_K}
                Q_K = \frac{(1-L) Q_\\mu}{Q_s / Q_p - 1}

            where (L = \frac{4}{3} \\left( \frac{\beta}{\alpha} \right)^2 \\), and (\\beta)
            and (\\alpha) are the shear and compressional wave velocities, respectively.

            Returns:
                Table: Anelastically corrected compressional wave speed table.
            """            # compute s and p wave velocities to compute "L".
            pwave_speed_table = super().compute_pwave_speed()
            swave_speed_table = super().compute_swave_speed()

            # Compute L
            L = 4 / 3 * (swave_speed_table.get_vals() / pwave_speed_table.get_vals()) ** 2

            # Meshing depths and temperatures of the anharmonic model to get all combinations
            depths_x, temperatures_x = numpy.meshgrid(
                pwave_speed_table.get_x(), pwave_speed_table.get_y(), indexing="ij")

            # computing Q_matrix for compressional wave
            Q_matrix_inv = (
                L / anelastic_model.compute_Q_shear(depths_x, temperatures_x) + (
                    1 - L) / anelastic_model.compute_Q_bulk(depths_x, temperatures_x)
            )

            # Apply anelastic correction
            corrected_vals = (
                pwave_speed_table.get_vals() * (1 - 0.5 / numpy.tan(anelastic_model.a(depths_x) * numpy.pi / 2) * Q_matrix_inv)
            )

            # return the anelastically corrected table
            return type(pwave_speed_table)(
                x=pwave_speed_table.get_x(),
                y=pwave_speed_table.get_y(),
                vals=corrected_vals,
                name=f"{pwave_speed_table.get_name()}"
            )

    return ThermodynamicModelPrime(
        thermo_model.model,
        thermo_model.composition,
        thermo_model.get_temperatures(),
        thermo_model.get_depths()
    )
