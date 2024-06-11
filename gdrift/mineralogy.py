import numpy
from pdb import set_trace as st
from .io import load_dataset
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize_scalar

MODELS_AVAIL = ['SLB_16']
COMPOSITIONS_AVAIL = ['pyrolite', 'basalt']


def LinearRectBivariateSpline(x, y, z):
    # This should be the case by default, but for some reason scipy does not catch this
    if not x.size == z.shape[0]:
        raise TypeError('x dimension of z must have same number of '
                        'elements as x')
    if not y.size == z.shape[1]:
        raise TypeError('y dimension of z must have same number of '
                        'elements as y')

    return RectBivariateSpline(x, y, z, kx=1, ky=1)


def dataset_name(model: str, composition: str):
    return f"{model}_{composition}"


class Table:
    """Base class for a table

        A table per definition has rows and columns
    """

    def __init__(self, x, y, vals, name=None):
        self._x = x
        self._y = y
        self._vals = vals
        self._name = name

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_vals(self):
        return self._vals

    def get_name(self):
        return self._name


class ThermodynamicModel(object):
    def __init__(self, model: str, composition: str, temps=None, depths=None):
        self.model = model
        self.composition = composition
        if model not in MODELS_AVAIL:
            raise ValueError(
                f"{model} not available. Use `print_available_models` to see all available models")

        # load the hdf5 table
        loaded_model = load_dataset(
            dataset_name(model, composition),
            table_names=["Depths", "Temperatures",
                         "bulk_mod", "shear_mod", "rho"]
        )
        # a dictionary that includes all the models
        self._tables = {}

        # the three tables that are needed
        for key in ["bulk_mod", "shear_mod", "rho"]:
            # in case we need to interpolate
            if any([temps is None, depths is None]):
                self._tables[key] = interpolate_table(
                    loaded_model["Depths"] if depths is None else depths,
                    loaded_model["Temperatures"] if temps is None else temps,
                    Table(
                        x=loaded_model.get("Depths"),
                        y=loaded_model.get("Temperatures"),
                        vals=loaded_model.get(key),
                        name=key)
                )
            else:
                self._tables[key] = Table(
                    x=loaded_model.get("Depths"),
                    y=loaded_model.get("Temperatures"),
                    vals=loaded_model.get(key),
                    name=key
                )

    def get_temperatures(self):
        return self._tables["shear_mod"].get_y()

    def get_depths(self):
        return self._tables["shear_mod"].get_x()

    def vs_to_temperature(self, vs, depth, bounds):
        vs_table = self.compute_swave_speed()
        bi_spline = LinearRectBivariateSpline(
            vs_table.get_x(),
            vs_table.get_y(),
            vs_table.get_vals)
        # TODO: pass in the bounds
        return numpy.array([self._find_temperature(v, d, bi_spline) for v, d in zip(vs, depth)])

    def vp_to_temperature(self, vp, depth):
        vp_table = self.compute_pwave_speed()
        bi_spline = LinearRectBivariateSpline(
            vp_table.get_x(),
            vp_table.get_y(),
            vp_table.get_vals)
        # TODO: pass in the bounds
        return numpy.array([self._find_temperature(v, d, bi_spline) for v, d in zip(vp, depth)])

    def temperature_to_vs(self, temperature, depth):
        vs = self.compute_swave_speed()
        return LinearRectBivariateSpline(
            vs.get_x(),
            vs.get_y(),
            vs.get_vals()).ev(depth, temperature)

    def temperature_to_vp(self, temperature, depth):
        vp = self.compute_pwave_speed()
        return LinearRectBivariateSpline(
            vp.get_x(),
            vp.get_y(),
            vp.get_vals).ev(depth, temperature)

    def temperature_to_rho(self, temperature, depth):
        return LinearRectBivariateSpline(
            self._tables["rho"].get_x(),
            self._tables["rho"].get_y(),
            self._tables["rho"].get_vals).ev(depth, temperature)

    def compute_swave_speed(self):
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
        return type(self._tables["shear_mod"])(
            x=self._tables["shear_mod"].get_x(),
            y=self._tables["shear_mod"].get_y(),
            vals=compute_pwave_speed(
                self._tables["bulk_mod"].get_vals(),
                self._tables["shear_mod"].get_vals(),
                self._tables["rho"].get_vals()),
            name="v_p")

    def _find_temperature(self, val, depth, interpolator, bounds=None):
        def objective(temp):
            return (interpolator(temp, depth) - val)**2
        result = minimize_scalar(objective, bounds=bounds, method='bounded')
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


def compute_pwave_speed(bulk_modulus, shear_modulus, density):
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
            bulk_modulus + (4./3.) * shear_modulus,
            density
        )
    )


def is_either_float_or_array(*args):
    if not all(isinstance(x, (float, numpy.ndarray)) for x in args):
        raise ValueError("All inputs must be either floats or numpy arrays.")

    if any(isinstance(x, numpy.ndarray) for x in args) and not all(isinstance(x, float) for x in args):
        if not all(x.shape == args[0].shape for x in args if isinstance(x, numpy.ndarray)):
            raise ValueError("All input arrays must have the same size.")
