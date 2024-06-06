from abc import ABC, abstractmethod
import numpy


class BaseAnelasticityModel(ABC):
    """
    Abstract base class for an anelasticity model.
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

    def __init__(self, B, g, a, solidus, omega=lambda x: 1.0):
        """
        Initialize the model with the given parameters.

        Args:
            B (function): Scaling factor for the Q model.
            g (function): Activation energy parameter.
            a (function): Frequency dependency parameter.
            solidus (function): Solidus temperature for mantle.
            omega (function): Seismic frequency (default is 1).
        """
        self.B = B
        self.g = g
        self.a = a
        self.omega = omega
        self.solidus = solidus

    def compute_Q(self, depths, temperatures, in_matrix_mode=True):
        """
        Computes the Q matrix based on depths and temperatures.

        Args:
            depths (numpy.ndarray): Depths at which the Q values are required.
            temperatures (numpy.ndarray): Temperatures corresponding to each depth.

        Returns:
            numpy.ndarray: A matrix of computed Q values.
        """
        depths = numpy.asarray(depths)
        temperatures = numpy.asarray(temperatures)

        Q_values = (self.B(depths) * (self.omega(depths)**self.a(depths))
                    * numpy.exp(
                        (self.a(depths) * self.g(depths)
                         * self.solidus.at_depth(depths)) / temperatures)
                    )
        return Q_values


def apply_anelastic_correction(thermo_model, anelastic_model):
    """
    Apply anelastic corrections to seismic velocity data using the provided anelasticity model.

    Args:
        thermo_model (ThermodynamicModel): The thermodynamic model with temperature and depth data.
        anelastic_model (BaseAnelasticityModel): An anelasticity model to compute Q values.

    Returns:
        Corrected ThermodynamicModel with anelastic effects.
    """
    class ThermodynamicModelPrime(thermo_model.__class__):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute_swave_speed(self):
            swave_speed_table = super().compute_swave_speed()
            depths_x, temperatures_x = numpy.meshgrid(
                swave_speed_table.get_x(), swave_speed_table.get_y(), indexing="ij")
            Q_matrix = anelastic_model.compute_Q(depths_x, temperatures_x)
            F = calculate_F(anelastic_model.a(depths_x))
            corrected_vals = swave_speed_table.get_vals(
            ) * (1 - (F / (numpy.pi * anelastic_model.a(depths_x))) * 1/Q_matrix)
            return type(swave_speed_table)(
                x=swave_speed_table.get_x(),
                y=swave_speed_table.get_y(),
                vals=corrected_vals,
                name=f"{swave_speed_table.get_name()}_anelastic_correction"
            )

        def compute_pwave_speed(self):
            pwave_speed_table = super().compute_pwave_speed()
            depths_x, temperatures_x = numpy.meshgrid(
                pwave_speed_table.get_x(), pwave_speed_table.get_y(), indexing="ij")
            Q_matrix = anelastic_model.compute_Q(depths_x, temperatures_x)
            F = calculate_F(anelastic_model.a(depths_x))
            corrected_vals = pwave_speed_table.get_vals(
            ) * (1 - (F / (numpy.pi * anelastic_model.a(depths_x))) * 1/Q_matrix)
            return type(pwave_speed_table)(
                x=pwave_speed_table.get_x(),
                y=pwave_speed_table.get_y(),
                vals=corrected_vals,
                name=f"{pwave_speed_table.get_name()}_anelastic_correction"
            )

    return ThermodynamicModelPrime(thermo_model.model, thermo_model.composition, thermo_model.get_temperatures(), thermo_model.get_depths())


def calculate_F(alpha):
    return ((numpy.pi*alpha)/2)*(1/numpy.tan(numpy.pi*alpha/2))
