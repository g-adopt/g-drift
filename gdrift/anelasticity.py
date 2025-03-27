from abc import ABC, abstractmethod
import numpy

# type hinting
from .profile import SplineProfile, RadialEarthModelFromFile, HirschmannSolidus
from .mineralogy import ThermodynamicModel
from typing import Type, TypeVar, Callable
import numpy.typing as npt

AnelasticityModel = TypeVar("AnelasticityModel", bound="BaseAnelasticityModel") # define a type hint for any subclass of `BaseAnelasticityModel`

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
            depths (npt.ArrayLike): Array of depths at which Q values are required.
            temperatures (npt.ArrayLike): Array of temperatures corresponding to the depths.

        Returns:
            npt.NDArray: A matrix of Q values corresponding to the given depths and temperatures.
        """
        pass

    @abstractmethod
    def compute_Q_bulk(self, depths: npt.ArrayLike, temperatures: npt.ArrayLike) -> npt.NDArray:
        """
        Computes the compressional anelastic quality factor (Q) matrix for given depths and temperatures.

        Args:
            depths (npt.ArrayLike): Array of depths at which Q values are required.
            temperatures (npt.ArrayLike): Array of temperatures corresponding to the depths.

        Returns:
            npt.NDArray: A matrix of Q values corresponding to the given depths and temperatures.
        """
        pass

    def build_ghelichkhan_solidus():
        # Load the solidus curve for the mantle from Andrault et al. (2011)
        andrault_solidus = RadialEarthModelFromFile(
            model_name="1d_solidus_Andrault_et_al_2011_EPSL",
            description="Andrault et al. 2011, EPSL")

        # Load the solidus curve for the mantle from Hirschmann (2000)
        hirsch_solidus = HirschmannSolidus()

        # Combining the two
        my_depths = []
        my_solidus = []
        for solidus_model in [hirsch_solidus, andrault_solidus.get_profile('solidus temperature')]:
            d_min, d_max = solidus_model.min_max_depth()
            dpths = numpy.arange(d_min, d_max, 10e3)
            my_depths.extend(dpths)
            my_solidus.extend(solidus_model.at_depth(dpths))

        # Avoding unnecessary extrapolation by setting the solidus temperature at maximum depth
        my_depths.extend([3000e3])
        my_solidus.extend([solidus_model.at_depth(dpths[-1])])

        # building the solidus profile that was originally used by Ghelichkhan et al. (2021)
        ghelichkhan_et_al = SplineProfile(
            depth=numpy.asarray(my_depths),
            value=numpy.asarray(my_solidus),
            extrapolate=True,
            name="Ghelichkhan et al. 2021")

        return ghelichkhan_et_al


class CammaranoAnelasticityModel(BaseAnelasticityModel):
    """
    A specific implementation of an anelasticity model following the approach by Cammarano et al. (2003).
    """

    def __init__(self,
                 B: Callable[[npt.ArrayLike], npt.NDArray],
                 g: Callable[[npt.ArrayLike], npt.NDArray],
                 a: Callable[[npt.ArrayLike], npt.NDArray],
                 omega: Callable[[npt.ArrayLike], npt.NDArray],
                 solidus: Type[SplineProfile],
                 Q_bulk: Callable[[npt.ArrayLike], npt.NDArray]):
        """
        Initialize the model with the given parameters.

        Args:
            B Callable[[npt.ArrayLike], Union[npt.NDArray, float]]: Grain size-related attenuation scaling factor.
            g Callable[[npt.ArrayLike], Union[npt.NDArray, float]]: Activation energy factor equal to $H(P)/RT_\text{m}(P)$ (Karato, 1993).
            a Callable[[npt.ArrayLike], Union[npt.NDArray, float]]: Exponent controlling frequency dependence.
            omega Callable[[npt.ArrayLike], Union[npt.NDArray, float]]: Seismic frequency.
            solidus (Type[RadialProfileSpline]): Solidus temperature for mantle.
            Q_bulk: Callable[[npt.ArrayLike], npt.NDArray]: Compressional anelastic quality factor.
        """
        self.B = B
        self.g = g
        self.a = a
        self.omega = omega
        self.solidus = solidus
        self.Q_bulk = Q_bulk

    @classmethod
    def from_q_profile(cls, q_profile: str):
        """
        Initialise the model with parameters corresponding to one of the Qn from Cammarano et al. (2003). This uses the solidus from Ghelichkhan et al. (2021).

        Args:
            q_profile (str): The name of the parameter set to use (e.g. "Q1" for Q1).
        """
        parameters = {
            "Q1": {
                "B": [0.5, 10],
                "g": [20, 10]
            },
            "Q2": {
                "B": [0.8, 15],
                "g": [20, 10]
            },
            "Q3": {
                "B": [1.1, 20],
                "g": [20, 10]
            },
            "Q4": {
                "B": [0.035, 2.25],
                "g": [30, 15]
            },
            "Q5": {
                "B": [0.056, 3.6],
                "g": [30, 15]
            },
            "Q6": {
                "B": [0.077, 4.95],
                "g": [30, 15]
            }
        }

        if q_profile not in parameters.keys():
            raise ValueError(f"Invalid argument: {q_profile}. Must be one of {parameters.keys()}")

        B = lambda x: numpy.where(x < 660e3, parameters[q_profile]["B"][0], parameters[q_profile]["B"][1])
        g = lambda x: numpy.where(x < 660e3, parameters[q_profile]["g"][0], parameters[q_profile]["g"][1])
        a = lambda x: 0.2 * numpy.ones_like(x)
        omega = lambda x: numpy.ones_like(x)
        solidus = cls.build_ghelichkhan_solidus()
        Q_bulk = lambda x: numpy.where(x < 660e3, 1e3, 1e4)
        
        return cls(B=B, g=g, a=a, omega=omega, solidus=solidus, Q_bulk=Q_bulk)

    def compute_Q_shear(self, depths: npt.ArrayLike, temperatures: npt.ArrayLike) -> npt.NDArray:
        """
        Compute the shear Q (attenuation quality factor) matrix based on input depths and temperatures.

        Args:
            depths (npt.ArrayLike): An array of depths at which Q values are to be calculated.
            temperatures (npt.ArrayLike): An array of temperatures corresponding to the specified depths.

        Returns:
            npt.NDArray: A matrix of calculated Q values, representing the shear attenuation quality factor.

        Notes:
            - If either `depths` or `temperatures` has a single element, it will be broadcasted.
            - For a full Q_shear matrix, `depths` and `temperatures` should be of compatible shapes (e.g., generated via `numpy.meshgrid`).
            - The computation uses the properties of the material, such as `B`, `omega`, `a`, and `g`, along with the solidus temperature at a given depth.

        Example:
            # Example usage:
            depths = numpy.linspace(0, 100, 50)  # 50 depth points from 0 to 100 km
            # Corresponding temperatures
            temperatures = numpy.linspace(800, 1200, 50)
            Q_matrix = model.compute_Q_shear(depths, temperatures)
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
        Computes the compressional anelastic quality factor (Q) matrix for given depths and temperatures.

        Args:
            depths (npt.ArrayLike): Array of depths at which Q values are required.
            temperatures (npt.ArrayLike): Array of temperatures corresponding to the depths.

        Returns:
            npt.NDArray: A matrix of Q values corresponding to the given depths and temperatures.

        Notes:
            - If either `depths` or `temperatures` has a single element, it will be broadcasted.
        """
        depths = numpy.asarray(depths)
        temperatures = numpy.asarray(temperatures)

        return self.Q_bulk(depths)
    
class GoesAnelasticityModel(BaseAnelasticityModel):
    """
    A specific implementation of an anelasticity model following the approach by Goes et al. (2004).
    """

    def __init__(self,
                 Q0: Callable[[npt.ArrayLike], npt.NDArray],
                 xi: Callable[[npt.ArrayLike], npt.NDArray],
                 a: Callable[[npt.ArrayLike], npt.NDArray],
                 omega: Callable[[npt.ArrayLike], npt.NDArray],
                 solidus: Type[SplineProfile],
                 Q_bulk: Callable[[npt.ArrayLike], npt.NDArray]):
        """
        Initialize the model with the given parameters.

        Args:
            Q0 Callable[[npt.ArrayLike], Union[npt.NDArray, float]]: Grain size-related attenuation scaling factor (equivalent to $B$ in Cammarano et al. [2003]).
            xi Callable[[npt.ArrayLike], Union[npt.NDArray, float]]: Activation energy factor equal to $aH(P)/RT_\text{m}(P)$ (Karato, 1993; equivalent to $ag$ in Cammarano et al. [2003]).
            a Callable[[npt.ArrayLike], Union[npt.NDArray, float]]: Exponent controlling frequency dependence.
            omega Callable[[npt.ArrayLike], Union[npt.NDArray, float]]: Seismic frequency.
            solidus (Type[RadialProfileSpline]): Solidus temperature for mantle.
            Q_bulk: Callable[[npt.ArrayLike], npt.NDArray]: Compressional anelastic quality factor.
        """
        self.Q0 = Q0
        self.xi = xi
        self.a = a
        self.omega = omega
        self.solidus = solidus
        self.Q_bulk = Q_bulk

    @classmethod
    def from_q_profile(cls, q_profile: str):
        """
        Initialise the model with parameters corresponding to one of the Qn from Goes et al. (2003). This uses the solidus from Ghelichkhan et al. (2021).

        Args:
            q_profile (str): The name of the parameter set to use (e.g. "Q4" for Q4).
        """
        parameters = {
            "Q4": {
                "Q0": [3.5, 35],
                "xi": [20, 10]
            },
            "Q6": {
                "Q0": [0.5, 3.5],
                "xi": [30, 20]
            }
        }

        if q_profile not in parameters.keys():
            raise ValueError(f"Invalid argument: {q_profile}. Must be one of {parameters.keys()}")

        Q0 = lambda x: numpy.where(x < 660e3, parameters[q_profile]["Q0"][0], parameters[q_profile]["Q0"][1])
        xi = lambda x: numpy.where(x < 660e3, parameters[q_profile]["xi"][0], parameters[q_profile]["xi"][1])
        a = lambda x: 0.15 * numpy.ones_like(x)
        omega = lambda x: numpy.ones_like(x)
        solidus = cls.build_ghelichkhan_solidus()
        Q_bulk = lambda x: numpy.where(x < 660e3, 1e3, 1e4)
        
        return cls(Q0=Q0, xi=xi, a=a, omega=omega, solidus=solidus, Q_bulk=Q_bulk)

    def compute_Q_shear(self, depths: npt.ArrayLike, temperatures: npt.ArrayLike) -> npt.NDArray:
        """
        Compute the shear Q (attenuation quality factor) matrix based on input depths and temperatures.

        Args:
            depths (npt.ArrayLike): An array of depths at which Q values are to be calculated.
            temperatures (npt.ArrayLike): An array of temperatures corresponding to the specified depths.

        Returns:
            npt.NDArray: A matrix of calculated Q values, representing the shear attenuation quality factor.

        Notes:
            - If either `depths` or `temperatures` has a single element, it will be broadcasted.
            - For a full Q_shear matrix, `depths` and `temperatures` should be of compatible shapes (e.g., generated via `numpy.meshgrid`).
            - The computation uses the properties of the material, such as `B`, `omega`, `a`, and `g`, along with the solidus temperature at a given depth.

        Example:
            # Example usage:
            depths = numpy.linspace(0, 100, 50)  # 50 depth points from 0 to 100 km
            # Corresponding temperatures
            temperatures = numpy.linspace(800, 1200, 50)
            Q_matrix = model.compute_Q_shear(depths, temperatures)
        """
        depths = numpy.asarray(depths)
        temperatures = numpy.asarray(temperatures)

        Q_values = (
            self.Q0(depths) * (self.omega(depths)**self.a(depths)) * numpy.exp(
                (self.a(depths) * self.xi(depths) * self.solidus.at_depth(depths)) / temperatures)
        )

        return Q_values

    def compute_Q_bulk(self, depths: npt.ArrayLike, temperatures: npt.ArrayLike) -> npt.NDArray:
        """
        Computes the compressional anelastic quality factor (Q) matrix for given depths and temperatures.

        Args:
            depths (npt.ArrayLike): Array of depths at which Q values are required.
            temperatures (npt.ArrayLike): Array of temperatures corresponding to the depths.

        Returns:
            npt.NDArray: A matrix of Q values corresponding to the given depths and temperatures.

        Notes:
            - If either `depths` or `temperatures` has a single element, it will be broadcasted.
        """
        depths = numpy.asarray(depths)
        temperatures = numpy.asarray(temperatures)

        return self.Q_bulk(depths)

def apply_anelastic_correction(thermo_model: Type[ThermodynamicModel], anelastic_model: Type[AnelasticityModel]):
    """
    Apply anelastic corrections to seismic velocity data using the provided "anelastic_model"
    within the low attenuation limit. The corrections are based on the equation
    $1 - \frac{V(anelastic)}{V(elastic)} = \frac{1}{2} \cot(\frac{\alpha \pi}{2}) Q^{-1},$
    as described by Stixrude & Lithgow-Bertelloni (doi:10.1029/2004JB002965, Eq-10).

        thermo_model (Type[ThermodynamicModel]): The thermodynamic model containing temperature and depth data.
        anelastic_model (Type[AnelasticityModel]): The anelasticity model containing 

    Returns:
        Type[ThermodynamicModel]: A new thermodynamic model with anelastically corrected seismic velocities.

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
