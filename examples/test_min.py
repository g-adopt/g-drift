import gdrift
from gdrift.profile import SplineProfile
import numpy as np


def build_thermodynamic_model(temperature_profile_array):
    # Load the thermodynamic model
    slb_pyrolite = gdrift.ThermodynamicModel("SLB_16", "pyrolite")

    # Make a spline that can be passed onto regularisation
    terra_temperature_spline = gdrift.SplineProfile(
        depth=temperature_profile_array[:, 0],
        value=temperature_profile_array[:, 1],
        name="T_average",
        extrapolate=True
    )

    # Regularise the thermodynamic model
    regular_slb_pyrolite = gdrift.regularise_thermodynamic_table(
        slb_pyrolite, terra_temperature_spline,
        regular_range={"v_s": (-1.0, 0.0), "v_p": (-np.inf, 0.0), "rho": (-np.inf, 0.0)})

    # building solidus model
    solidus_ghelichkhan = build_solidus()

    
    # Using the solidus model build the anelasticity model around the solidus profile
    anelasticity = build_anelasticity_model(solidus_ghelichkhan)
    # Apply the anelasticity correction to the regularised thermodynamic model
    anelastic_slb_pyrolite = gdrift.apply_anelastic_correction(
        regular_slb_pyrolite, anelasticity)

    return anelastic_slb_pyrolite


# Compute a solidus for building anelasticity correction
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


def get_dimensional_parameters():
    return {
        "T_CMB": 4000.0,
        "T_surface": 300.0,
        "rho": 3200.0,
        "g": 9.81,
        "cp": 1249.7,
        "alpha": 4.1773e-05,
        # "kappa": 3.0,
        # "H_int": 2900e3,
    }


dpth, T_ave = np.loadtxt("./profile.txt", delimiter=",", unpack=True)

slb_model = build_thermodynamic_model(np.column_stack((dpth, T_ave)))
