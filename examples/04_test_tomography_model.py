import gdrift
import numpy as np
from gdrift.profile import SplineProfile
import matplotlib.pyplot as plt


def __compare_loaded_model__():

    prem = gdrift.PreliminaryRefEarthModel()

    tdmodel = gdrift.ThermodynamicModel("SLB_16", "pyrolite", temps=None, depths=None)
    tdmodel_vs = tdmodel.compute_swave_speed().get_vals()
    slb_dataset = gdrift.load_dataset("SLB_16_pyrolite")
    depths = slb_dataset["Depths"]
    v_s = slb_dataset["v_s"]

    plt.close(1)
    fig = plt.figure(num=1)
    ax = fig.add_subplot(111)
    for depth in [100e3]:
        index = abs(depths - depth).argmin()
        ax.plot(slb_dataset["Temperatures"], v_s[index, :], label=f"Depth: {depth / 1e3:.1f} km")
        ax.plot(tdmodel.get_temperatures(), tdmodel_vs[index, :] / 1e3, linestyle="-")
        ax.axhline(prem.at_depth("Vsh", depth) / 1e3, linestyle='--')

    ax.legend()
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("S-wave speed (km/s)")
    plt.show()


def __main__load__():
    # Load the preliminary reference Earth model (PREM)
    prem = gdrift.PreliminaryRefEarthModel()

    target_depth = 410e3

    slb_dataset = gdrift.load_dataset("SLB_16_pyrolite")
    temperature = slb_dataset["Temperatures"]
    depths = slb_dataset["Depths"]

    index = abs(depths - target_depth).argmin()

    v_s = slb_dataset["v_s"]

    plt.close(1)
    fig = plt.figure(num=1)
    ax = fig.add_subplot(111)
    ax.plot(temperature, v_s[index, :])
    ax.text(0.5, 1.0, f"Depth: {depths[index] / 1e3:.1f} km", transform=ax.transAxes)
    ax.axhline(prem.at_depth("Vsh", depths[index]) / 1e3)
    fig.show()


def __main__():
    # Load the preliminary reference Earth model (PREM)
    prem = gdrift.PreliminaryRefEarthModel()

    # some depths to evaluate the model at
    depths = np.linspace(100e3, 2890e3, 20)

    # Extract the values of the S-wave speed at the specified depths
    vs_values = prem.at_depth("Vsh", depths)

    # Get shear seismic speeds from model
    slb_pyrolite_anelastic = build_thermodynamic_model(None)
    prem_temperature = slb_pyrolite_anelastic.vs_to_temperature(vs_values, depths)

    table_vs = slb_pyrolite_anelastic.compute_swave_speed()

    plt.close(2)
    fig = plt.figure(num=2)
    ax = fig.add_subplot(111)
    index = 2
    for index in range(0, 20):
        # ax.plot(table_vs.get_y(), table_vs.get_vals()[index, :], label="Regularised SLB16 Pyrolite")
        ax.plot(table_vs.get_y(), table_vs.get_vals()[abs(table_vs.get_x() - depths[index]).argmin(), :], label=f"Depth: {depths[index] / 1e3:.1f} km")
        ax.scatter(prem_temperature[index], vs_values[index], label="PREM")
    fig.show()


def build_solidus():
    # Defining the solidus curve for manlte
    andrault_solidus = gdrift.RadialEarthModelFromFile(
        model_name="1d_solidus_Andrault_et_al_2011_EPSL",
        description="Andrault et al 2011 EPSL")

    # Defining parameters for Cammarano style anelasticity model
    hirsch_solidus = gdrift.HirschmannSolidus()

    my_depths = []
    my_solidus = []
    for solidus_model in [hirsch_solidus, andrault_solidus]:
        d_min, d_max = solidus_model.min_max_depth("solidus temperature")
        dpths = np.arange(d_min, d_max, 10e3)
        my_depths.extend(dpths)
        my_solidus.extend(solidus_model.at_depth("solidus temperature", dpths))

    ghelichkhan_et_al = SplineProfile(
        depth=np.asarray(my_depths),
        value=np.asarray(my_solidus),
        name="Ghelichkhan et al 2021",
        extrapolate=True)

    return ghelichkhan_et_al


def build_anelasticity_model(solidus):
    def B(x):
        return np.where(x < 660e3, 1.1, 20)

    def g(x):
        return np.where(x < 660e3, 20, 10)

    def a(x):
        return 0.2

    def omega(x):
        return 1.0

    return gdrift.CammaranoAnelasticityModel(B, g, a, solidus, omega)


def build_thermodynamic_model(depths):
    # Thermodynamic model
    slb_pyrolite = gdrift.ThermodynamicModel("SLB_16", "pyrolite", temps=None, depths=depths)

    # The solid phase changes for mantle minerals are a source of error when
    # interpreting seismic tomography speed in terms of temperature. This
    # has been explain in detail in Ghelichkhan et al 2021. Here we regularise
    # the thermodynamic table along a specific temperature profile. The regularisaion
    # is done by limiting derivatives with respect to temperature, and anchoring the
    # curves to the original value of vs/T at each depth.

    # We first load the temperature profile
    # first column contains depths, and second column contains temperatures
    terra_temperature_array = np.loadtxt("TerraMT512vs.dat", unpack=False, usecols=(0, 1))
    # Make a spline that can be passed onto regularisation
    terra_temperature_spline = gdrift.SplineProfile(
        depth=terra_temperature_array[:, 0] * 1e3,
        value=terra_temperature_array[:, 1],
        name="TerraMT512vs",
        extrapolate=True
    )

    # regularise the thermodynamic table using default values
    # Default values only prohibit positive jumps in the derivative
    # These jumps are associated with phase changes, and make the
    # problem of finding the associated temperature with a certain
    # seismic speed at depths such as 660 km non-unique.
    regular_slb_pyrolite = gdrift.regularise_thermodynamic_table(
        slb_pyrolite, terra_temperature_spline,
        regular_range={"v_s": (-1.5, 0.0), "v_p": (-np.inf, 0.0), "rho": (-np.inf, 0.0)})

    # building solidus model
    solidus_ghelichkhan = build_solidus()
    anelasticity = build_anelasticity_model(solidus_ghelichkhan)
    anelastic_regular_slb_pyrolite = gdrift.apply_anelastic_correction(
        regular_slb_pyrolite, anelasticity)

    return anelastic_regular_slb_pyrolite


if __name__ == "__main__":
    __main__()
    __main__load__()
    __compare_loaded_model__()
