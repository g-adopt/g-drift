import numpy
from .io import load_dataset
from scipy.spatial import cKDTree

MODELS_AVAIL = ['SLB_16']
COMPOSITIONS_AVAIL = ['pyrolite', 'basalt']


def dataset_name(model: str, composition: str):
    return f"{model}_{composition}"


def print_available_models():
    print(f"All available models {MODELS_AVAIL}")


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
    def __init__(self, temps, depths, model: str, composition: str):
        if model not in MODELS_AVAIL:
            raise ValueError(
                f"{model} not available."
                " Use `print_available_models` to see all available models")

        loaded_model = load_dataset(
            dataset_name(model, composition),
            table_names=["Depths", "Temperatures",
                         "bulk_mod", "shear_mod", "density"]
        )
        # setting up a dictionary that includes all the models
        self._tables = {}
        for key in ["bulk_mod", "shear_mod", "density"]:
            self._tables[key] = interpolate_table(
                temps,
                depths,
                Table(
                    x=loaded_model.get("Temperatures"),
                    y=loaded_model.get("Depths"),
                    vals=loaded_model.get(key),
                    name=key)
            )


def interpolate_table(ox, oy, table_in):
    def min_max(v):
        return numpy.min(v), numpy.max(v)

    x_x, y_x = numpy.meshgrid(table_in.get_x(), table_in.get_y())

    minx, maxx = min_max(x_x)
    miny, maxy = min_max(y_x)

    x_x = (x_x - minx)/(maxx - minx)
    y_x = (y_x - miny)/(maxy - miny)

    tree = cKDTree(numpy.column_stack((x_x.flatten(), y_x.flatten())))

    ox_x, oy_x = numpy.meshgrid(ox, oy)

    ox_x = (ox_x - minx)/(maxx - minx)
    oy_x = (oy_x - miny)/(maxy - miny)

    dists, inds = tree.query(numpy.column_stack(
        (ox_x.flatten(), oy_x.flatten())), k=2)

    ovals = numpy.einsum(
        "i, i",
        numpy.einsum("ij->i", 1/dists),
        numpy.einsum("ij, ij -> i",
                     1/dists,
                     table_in.get_vals().flatten()[inds])
    )
    return type(table_in)(ox, oy, ovals, name=table_in.get_name())


def compute_swave_speed(shear_modulus, density):
    """ Calculate the S-wave (secondary or shear wave) speed in a material based on its
    shear modulus and density. Inputs can be floats or numpy arrays of the same size.

    Args:
        shear_modulus (float or np.ndarray): The shear modulus of the material,
            indicating its resistance to shear deformation.
        density (float or np.ndarray): The density of the material

    Returns:
        float or np.ndarray: The speed of S-waves in the material, calculated in meters
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
        bulk_modulus (float or np.ndarray): The bulk modulus of the material, representing its resistance
            to uniform compression. Typically measured in pascals (Pa).
        shear_modulus (float or np.ndarray): The shear modulus of the material, indicating its resistance
            to shear deformation. It is also typically measured in pascals (Pa).
        density (float or np.ndarray): The density of the material, measured in kilograms per cubic meter (kg/m^3).

    Returns:
        float or np.ndarray: The speed of P-waves in the material, calculated in meters per second (m/s).
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