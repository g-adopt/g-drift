import numpy
from .profile import AbstractProfile
from .io import load_dataset
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from numbers import Number
from typing import Optional, Tuple, Union, Dict
import numpy as np

# Default regular range for gradients
# This will be used in regularise_thermodynamic_table
# if nothing is provided
default_regular_range = {
    "v_s": (-np.inf, 0.0),
    "v_p": (-np.inf, 0.0),
    "rho": (-np.inf, 0.0),
}


MODELS_AVAIL = ['SLB_16', "SLB_21"]
COMPOSITIONS_AVAIL = ['pyrolite', 'basalt']


def LinearRectBivariateSpline(x, y, z):

    # This should be the case by default, but for some reason scipy does not catch this
    if not x.size == z.shape[0]:
        raise TypeError('x dimension of z must have same number of '
                        'elements as x')
    if not y.size == z.shape[1]:
        raise TypeError('y dimension of z must have same number of '
                        'elements as y')

    return RectBivariateSpline(
        x, y, z,
        bbox=[x[0], x[-1], y[0], y[-1]],
        kx=1, ky=1)


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

        # Todo: I am commenting this out, but it should be replace in load_dataset
        # if model not in MODELS_AVAIL:
        #     raise ValueError(
        #         f"{model} not available. Use `print_available_models` to see all available models")

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
            if temps is not None or depths is not None:
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

    def vs_to_temperature(self, vs: Number, depth: Number, bounds: Optional[Union[Tuple[float, float], Tuple[numpy.ndarray, numpy.ndarray]]] = (300, 7000)) -> Number:
        """
        Convert S-wave velocity (vs) to temperature at a given depth.
        Parameters:
        -----------
        vs : Number
            The S-wave velocity.
        depth : Number
            The depth at which the temperature is to be calculated.
        bounds : Optional[Union[Tuple[float, float], Tuple[numpy.ndarray, numpy.ndarray]]], default=(300, 7000)
            The bounds for the temperature calculation. It can be a tuple of floats or numpy arrays.
        Returns:
        --------
        Number
            The temperature corresponding to the given S-wave velocity and depth.
        """
        return self._v_to_temperature(vs, depth, self.compute_swave_speed(), bounds)

    def vp_to_temperature(self, vp: Number, depth: Number, bounds: Optional[Union[Tuple[float, float], Tuple[numpy.ndarray, numpy.ndarray]]] = (300, 7000)) -> Number:
        """
        Convert P-wave velocity (vp) to temperature at a given depth.
        Parameters:
        -----------
        vp : Number
            The P-wave velocity.
        depth : Number
            The depth at which the temperature is to be calculated.
        bounds : Optional[Union[Tuple[float, float], Tuple[numpy.ndarray, numpy.ndarray]]], default=(300, 7000)
            The bounds for the temperature calculation. It can be a tuple of floats or numpy arrays.
        Returns:
        --------
        Number
            The temperature corresponding to the given P-wave velocity and depth.
        """
        return self._v_to_temperature(vp, depth, self.compute_pwave_speed(), bounds)

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
            vp.get_vals()).ev(depth, temperature)

    def temperature_to_rho(self, temperature, depth):
        return LinearRectBivariateSpline(
            self._tables["rho"].get_x(),
            self._tables["rho"].get_y(),
            self._tables["rho"].get_vals()).ev(depth, temperature)

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

    def _v_to_temperature(self,
                          v: Number,
                          depth: Number,
                          table: Table,
                          bounds: Optional[Union[Tuple[float, float], Tuple[numpy.ndarray, numpy.ndarray]]] = (300, 7000)) -> Number:
        """
        Convert any wave speed to temperature at given depths deping on the table provided.

        Parameters:
        v (Number): wave speed.
        depth (Number): Depth at which the temperature is to be calculated.
        bounds (Optional[Union[Tuple[float, float], Tuple[numpy.ndarray, numpy.ndarray]]]):
            Bounds for the temperature search. If not provided, default bounds [300, 7000] are used.

        Returns:
        numpy.ndarray: Temperature corresponding to the given wave speed and depth.
        """

        # check if bounds is a tuple of floats
        if isinstance(bounds, tuple) and all(isinstance(b, (float, int)) for b in bounds):
            bounds = tuple([numpy.full_like(v, b) for b in bounds])

        # Ensure vs, depth, and bounds are all of the same shape
        if not (v.shape == depth.shape == bounds[0].shape == bounds[1].shape):
            raise ValueError("vs, depth, and bounds must all have the same shape")

        # create a bivariate spline for the table as interpolater
        bi_spline = LinearRectBivariateSpline(
            table.get_x(),
            table.get_y(),
            table.get_vals())

        # return the temperature
        return numpy.squeeze(
            numpy.array(
                [self._find_temperature(a_speed, a_depth, bi_spline, bounds=(lb, ub)) for a_speed, a_depth, lb, ub in zip(v, depth, bounds[0], bounds[1])]
            )
        )

    def _find_temperature(self, val, depth, interpolator, bounds):
        def objective(temp):
            return (interpolator(depth, temp) - val)**2

        result = minimize_scalar(
            objective,
            bounds=[bounds[0], bounds[1]],
            method='bounded',
            options={'xatol': 1e-2}
        )
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


def compute_pwave_speed(bulk_modulus: Number, shear_modulus: Number, density: Number) -> Number:
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
            bulk_modulus + (4. / 3.) * shear_modulus,
            density
        )
    )


def is_either_float_or_array(*args):
    if not all(isinstance(x, (float, numpy.ndarray)) for x in args):
        raise ValueError("All inputs must be either floats or numpy arrays.")

    if any(isinstance(x, numpy.ndarray) for x in args) and not all(isinstance(x, float) for x in args):
        if not all(x.shape == args[0].shape for x in args if isinstance(x, numpy.ndarray)):
            raise ValueError("All input arrays must have the same size.")


def derive_then_integrate(table: Table, temperature_profile: AbstractProfile, regular_range: Dict[str, Tuple]) -> np.ndarray:
    """
    Derives the temperature gradient, interpolates irregular values, and integrates again to obtain velocity.
    The output is anchored (= 0.) at around velocity values that are associated at temperature_profile.
    Args:
        table (object): An object containing depth and temperature data with methods `get_x()`, `get_y()`, and `get_vals()`.
        temperature_profile (object): An object with a method `at_depth(depths)` that returns temperature values at given depths.
        regular_range (dict): A dictionary with keys corresponding to table names and values as tuples indicating the acceptable range for gradients.
    Returns:
        np.ndarray: A 2D array representing the integrated velocity values adjusted for the temperature profile.
    """

    # Getting the name of the table
    key = table._name
    # Getting the depths and temperatures
    depths = table.get_x()
    temperatures = table.get_y()

    # temperature gradient
    dT = np.gradient(temperatures)

    # Creating a mesh for the depths and temperatures
    depths_x, temperatures_x = np.meshgrid(depths, temperatures, indexing="ij")

    # Getting the gradients
    dV_dT = np.gradient(table.get_vals(), depths, temperatures, axis=(0, 1))[1]

    # Finding the regular range of values (No positive jumps, no high negative jumps)
    within_range = np.logical_and(dV_dT < regular_range[key][1], dV_dT > regular_range[key][0])

    # building a tree out of the regular values
    my_tree = cKDTree(np.column_stack((depths_x[within_range].flatten(), temperatures_x[within_range].flatten())))

    # Finding the closest values to the irregular values
    distances, inds = my_tree.query(np.column_stack((depths_x[~ within_range].flatten(), temperatures_x[~ within_range].flatten())), k=3)

    # Interpolating the irregular values
    dV_dT[~within_range] = np.sum(1 / distances * dV_dT[within_range].flatten()[inds], axis=1) / np.sum(1 / distances, axis=1)

    # Integrating the derivate again to get the velocity (note that a constant needs to be found)
    V = np.cumsum(dV_dT * dT, axis=1)
    # One D profile of vs that best describes the temperature profile
    t_mean_array = np.asarray([V[i, j] for i, j in enumerate(abs(temperature_profile.at_depth(depths_x) - temperatures).argmin(axis=1))])

    # Broadcasting to the correct shape
    t_mean_array_x, _ = np.meshgrid(t_mean_array, temperatures, indexing="ij")

    # Anchoring the V-T curve at each depth for acnhor T to have zero velocity
    V -= t_mean_array_x

    return V


def regularise_thermodynamic_table(slb_pyrolite: ThermodynamicModel, temperature_profile: AbstractProfile, regular_range: Dict[str, Tuple] = default_regular_range):
    """
    Regularises the thermodynamic table by creating a regularised thermodynamic model that uses precomputed
    regular tables for S-wave and P-wave speeds.

    Args:
        slb_pyrolite (ThermodynamicModel): The original thermodynamic model.
        temperature_profile (AbstractProfile): The temperature profile to be used for regularisation. This is supposed to
            be a 1D profile of average temperature profiles.
        regular_range (Dict[str, Tuple], optional): Dictionary specifying the regularisation range for each
            parameter. Defaults to `gdrift.mineralogy.default_regular_range`.

    Returns:
        RegularisedThermodynamicModel: A regularised thermodynamic model with precomputed tables for S-wave
        and P-wave speeds.
    """
    # regular tables are a dictaionary of tables
    regular_tables = {}

    # iterating over the tables
    for table, convert_T2V in zip([slb_pyrolite._tables["rho"], slb_pyrolite.compute_swave_speed(), slb_pyrolite.compute_pwave_speed()],
                                  [slb_pyrolite.temperature_to_rho, slb_pyrolite.temperature_to_vs, slb_pyrolite.temperature_to_vp]):
        # Get name for the table
        key = table._name

        regular_tables[key] = derive_then_integrate(table, temperature_profile, regular_range)

        # the velocity for the given temperature profile
        v_average = convert_T2V(temperature=temperature_profile.at_depth(table.get_x()), depth=table.get_x())

        # Subtracting the mean
        regular_tables[key] += v_average[:, None]

    class RegularisedThermodynamicModel(ThermodynamicModel):
        """
        A wrapper class for a regularised thermodynamic model that uses precomputed regular tables
        for S-wave and P-wave speed instead of the default methods.
        """

        def __init__(self, *args, **kwargs):
            # Inherit properties from the original model
            super().__init__(*args, **kwargs)
            self._tables["rho"] = Table(self.get_depths(), self.get_temperatures(), regular_tables["rho"], name="rho")

        def compute_swave_speed(self):
            """
            Returns the regularised S-wave speed as a `Table` object.
            """
            return Table(self.get_depths(), self.get_temperatures(), regular_tables["v_s"], name="v_s")

        def compute_pwave_speed(self):
            """
            Returns the regularised P-wave speed as a `Table` object.
            """
            return Table(self.get_depths(), self.get_temperatures(), regular_tables["v_p"], name="v_p")

    return RegularisedThermodynamicModel(
        slb_pyrolite.model,
        slb_pyrolite.composition,
        slb_pyrolite.get_temperatures(),
        slb_pyrolite.get_depths())
