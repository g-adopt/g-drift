import numpy as np
import gdrift
from gdrift.profile import RadialProfileSpline


def build_solidus():
    # Defining the solidus curve for manlte
    andrault_solidus = gdrift.SolidusProfileFromFile(
        model_name="1d_solidus_Andrault_et_al_2011_EPSL",
        description="Andrault et al 2011 EPSL")

    # Defining parameters for Cammarano style anelasticity model
    hirsch_solidus = gdrift.HirschmannSolidus()

    my_depths = []
    my_solidus = []
    for solidus_model in [hirsch_solidus, andrault_solidus]:
        d_min, d_max = solidus_model.min_max_depth()
        dpths = np.arange(d_min, d_max, 10e3)
        my_depths.extend(dpths)
        my_solidus.extend(solidus_model.at_depth(dpths))

    ghelichkhan_et_al = RadialProfileSpline(
        depth=np.asarray(my_depths),
        value=np.asarray(my_solidus),
        name="Ghelichkhan et al 2021")

    return ghelichkhan_et_al


# Load PREM
prem = gdrift.PreRefEarthModel()

# Thermodynamic model
slb_pyrolite = gdrift.ThermodynamicModel("SLB_16", "pyrolite")

# building solidus model
solidus_ghelichkhan = build_solidus()
