from abc import ABC, abstractmethod
import numpy


class BaseAnelasticityModel(ABC):
    """
    Abstract base class for an anelasticity model.
    All anelasticity models must be able to compute a Q matrix given depths and temperatures.
    """

    @abstractmethod
    def compute_Q(self, depths, temperatures):
        """
        Computes the anelastic quality factor (Q) matrix for given depths and temperatures.

        Args:
            depths (numpy.ndarray): Array of depths at which Q values are required.
            temperatures (numpy.ndarray): Array of temperatures corresponding to the depths.

        Returns:
            numpy.ndarray: A matrix of Q values corresponding to the given depths and temperatures.
        """
        pass


class CammaranoAnelasticityModel(BaseAnelasticityModel):
    """
    A specific implementation of an anelasticity model following the approach by Cammarano et al.
    """

    def __init__(self, B, g, a, omega=1):
        """
        Initialize the model with the given parameters.

        Args:
            B (float): Scaling factor for the Q model.
            g (float): Activation energy parameter.
            a (float): Frequency dependency parameter.
            omega (float): Seismic frequency (default is 1).
        """
        self.B = B
        self.g = g
        self.a = a
        self.omega = omega

    def compute_Q(self, depths, temperatures):
        """
        Computes the Q matrix based on depths and temperatures.

        Args:
            depths (numpy.ndarray): Depths at which the Q values are required.
            temperatures (numpy.ndarray): Temperatures corresponding to each depth.

        Returns:
            numpy.ndarray: A matrix of computed Q values.
        """
        Q_values = self.B * (self.omega**self.a) * numpy.exp((self.a * self.g * 1600) /
                                                          temperatures)  # 1600 is a placeholder for a reference temperature
        return Q_values


def apply_anelastic_correction(thermo_model, anelastic_model):
    """
    Apply anelastic corrections to seismic velocity data using the provided anelasticity model.

    Args:
        thermo_model (ThermodynamicModel): The thermodynamic model with temperature and depth data.
        anelastic_model (BaseAnelasticityModel): An anelasticity model to compute Q values.

    Returns:
        Corrected anelastic effect
    """
    class ThermodynamicModelPrime(thermo_model):
        def compute_swave_speed(self):
            # Get the original s-wave speed table
            swave_speed_table = super().compute_swave_speed()
            # Apply the post-processing function
            # build a Q matrix for given depths and temperature
            Q_matrix = anelastic_model.compute_Q(
                swave_speed_table.get_depth(),
                swave_speed_table.get_temperature()
            )

            F = calculate_F(anelastic_model.alpha)  # Calculate the F factor based on alpha
            corrected_vals = swave_speed_table.get_vals() * (1 - (F / (numpy.pi * anelastic_model.alpha)) * 1/Q_matrix )

            # Return a new table with modified values but same x and y
            return type(swave_speed_table)(
                x=swave_speed_table.get_x(),
                y=swave_speed_table.get_y(),
                vals=corrected_vals,
                name=swave_speed_table.get_name()
            )
    return ThermodynamicModelPrime


def calculate_F(alpha):
    """
    Calculate the factor F based on alpha for the anelastic correction.

    Args:
        alpha (numpy.ndarray): Array of alpha values derived from the Q matrix.

    Returns:
        numpy.ndarray: Calculated F values for each alpha.
    """
    F = ((numpy.pi * alpha) / 2) * (1 / numpy.tan(numpy.pi * alpha / 2))
    return F
